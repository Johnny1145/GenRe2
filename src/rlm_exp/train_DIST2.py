import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import rootutils
import swanlab
import torch
from accelerate import Accelerator
try:
    from accelerate import DeepSpeedPlugin
except Exception:
    DeepSpeedPlugin = None
from accelerate.utils import tqdm as accelerate_tqdm
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.checkpoint import CheckpointManager
from src.utils.rlm_DIST2 import DIST2Loss
from src.utils.reinforce_loss import ReinforceLoss

from ..model.base_module import BaseModule
from ..model.regress_lm import core
from ..model.regress_lm.models.pytorch import model as torch_model_lib
from ..model.regress_lm.tokenizers import P10Tokenizer
from ..model.regress_lm.vocabs import DecoderVocab, SentencePieceVocab
from src.data.dataset.rlm_dataset import RLMDataset

# Initialize vocabs
encoder_vocab = SentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=P10Tokenizer())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_trainable_parameters(model: torch.nn.Module):
    """

    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            #
            print(f"  - Trainable: {name}")

    trainable_percentage = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable %: {trainable_percentage:.2f}"
    )

def freeze_encoder_parameters(model_adapter):
    """
     HuggingFaceRegressionAdapter encoder

    Args:
        model_adapter (HuggingFaceRegressionAdapter):  Hugging Face
    """
    logger.info("Freezing encoder parameters...")
    frozen_count = 0
    total_params_in_frozen_tensors = 0

    for name, param in model_adapter.named_parameters():
        # ""'model.encoder'
        #  'model.model.model.encoder...'
        if 'model.encoder' in name:
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
                total_params_in_frozen_tensors += param.numel()

    if frozen_count > 0:
        logger.info(f"Successfully froze {frozen_count} parameter tensors in the encoder, totaling {total_params_in_frozen_tensors:,} parameters.")
    else:
        logger.warning("Could not find any parameters to freeze containing '.model.encoder'. Check parameter names.")

class HuggingFaceRegressionAdapter(torch.nn.Module):
    """ Hugging Face

     cfg.hf
      - enabled: bool
      - model_name_or_path: str
      - trust_remote_code: bool ()
    """

    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None):
        super().__init__()
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                " transformers  Hugging Face : " + str(e)
            )

        # HuggingFace Hub
        # os.environ["HF_HUB_OFFLINE"] = "1"

        local_dir = getattr(cfg.hf, "local_dir", None)
        model_name_or_path = local_dir if local_dir else cfg.hf.model_name_or_path
        trust_remote_code = getattr(cfg.hf, "trust_remote_code", True)  # trust_remote_code
        print(f"Loading model from: {model_name_or_path}")

        # HuggingFace Hubtest_rlm.py
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,  #
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,  #
        )

        # RLM
        self.n_out_tokens = getattr(self.model.config, "num_tokens_per_obj", 8) * getattr(self.model.config, "max_num_objs", 1)
        if hasattr(self.model.config, "num_tokens_per_obj") and hasattr(self.model.config, "max_num_objs"):
            self.n_out_tokens = self.model.config.num_tokens_per_obj * self.model.config.max_num_objs

        self.decoder_vocab = None
        self.device = device if device is not None else torch.device("cpu")
        self.DIST2 = DIST2Loss(self.tokenizer, self.device)
        self.to(self.device)

    def forward(self, **inputs):
        return self.model.generate(**inputs)

    def convert_numeric_examples(self, examples: List[Dict]):
        """CE labels"""
        batch_inputs = {}
        if len(examples) == 0:
            return {}

        texts = []
        targets = []

        # ==================== [] ====================
        #
        for ex in examples:
            # RLM
            if hasattr(ex, 'input') and hasattr(ex, 'target'):
                text = ex.input
                target = float(ex.target)
            elif hasattr(ex, 'x') and hasattr(ex, 'y'):
                text = ex.x
                target = float(ex.y)
            else:
                #
                text = ex.get('input', ex.get('x', ''))
                target = float(ex.get('target', ex.get('y', 0)))

            # []  append
            #
            texts.append(text)
            targets.append(target)
        # ======================================================

        # tokenizer
        #  texts padding=True
        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        batch_inputs.update(enc)

        # target
        batch_inputs["y"] = torch.tensor(targets, dtype=torch.float32)

        # CE labelstoken
        labels_list = []
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        for t in targets:
            ids = self.tokenizer.float_to_token_ids(float(t))
            # /
            if len(ids) < self.n_out_tokens:
                ids = ids + [pad_id] * (self.n_out_tokens - len(ids))
            else:
                ids = ids[:self.n_out_tokens] #
            labels_list.append(ids)
        batch_inputs["labels"] = torch.tensor(labels_list, dtype=torch.long)

        #  labels_list  decoder_input_ids teacher forcing  decoder_start_token_id
        start_id = (
            getattr(self.model.config, "decoder_start_token_id", None)
            if hasattr(self.model, "config") else None
        )
        if start_id is None:
            start_id = getattr(self.tokenizer, "bos_token_id", None)
            if start_id is None:
                start_id = pad_id

        decoder_input_ids_list = []
        for ids in labels_list:
            if len(ids) == 0:
                decoder_ids = [start_id]
            else:
                decoder_ids = [start_id] + ids[:-1]
            decoder_input_ids_list.append(decoder_ids)

        decoder_input_ids = torch.tensor(decoder_input_ids_list, dtype=torch.long)
        decoder_attention_mask = (decoder_input_ids != pad_id).long()

        batch_inputs["decoder_input_ids"] = decoder_input_ids
        batch_inputs["decoder_attention_mask"] = decoder_attention_mask

        return batch_inputs

    def compute_loss_and_metrics(self, batch: Dict):
        """
        RLM generate
        """
        # 1.
        targets = batch.get("y")
        if targets is None:
            raise ValueError(" batch['y'] ")
        targets = targets.to(self.device)

        #  decoder_input_ids
        #  'labels'
        allowed_input_keys = ["input_ids", "attention_mask", "decoder_input_ids"]
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in allowed_input_keys}

        #  decoder_input_ids
        labels_for_loss = batch.get("labels").to(self.device)
        if "decoder_input_ids" not in inputs:
            decoder_start_token_id = self.model.config.decoder_start_token_id
            start_tokens = torch.full((labels_for_loss.size(0), 1), decoder_start_token_id, dtype=torch.long, device=labels_for_loss.device)
            shifted_labels = labels_for_loss[:, :-1]
            inputs["decoder_input_ids"] = torch.cat([start_tokens, shifted_labels], dim=-1)

        try:

            # ==================== [] ====================
            # a.  'labels'
            outputs = self.model(**inputs, labels=None)

            # b.  logits [batch_size, seq_len, decoder_vocab_size]
            logits = outputs.logits

            # c.
            loss_fct = torch.nn.CrossEntropyLoss() #  index  -100  target

            # #  logits  labels
            # # logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # # labels: [batch_size, seq_len] -> [batch_size * seq_len]

            # #  labels for loss padding token  -100
            loss_labels = labels_for_loss.clone()
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            loss_labels[loss_labels == pad_token_id] = -100

            ce_loss = loss_fct(logits.view(-1, logits.size(-1)), loss_labels.view(-1))
            DIST2_loss = self.DIST2.forward(logits, labels_for_loss)
            loss = ce_loss + 0.1 * DIST2_loss
            # ======================================================

        except Exception as e:
            logger.error(f"")
            logger.error(f"Inputs to model: {list(inputs.keys())}")
            if 'outputs' in locals() and hasattr(outputs, 'logits'):
                logger.error(f"  - outputs.logits shape: {outputs.logits.shape}")
            if 'labels_for_loss' in locals():
                logger.error(f"  - labels_for_loss shape: {labels_for_loss.shape}")
            raise e


        # ---  ---
        #  model.generate

        return loss, {
            "ce_loss": float(ce_loss.detach().item()),
            "DIST2_loss": float(DIST2_loss.detach().item()),
            "loss": float(loss.detach().item()),
        }

    def load_from_pretrained_dir(self, dir_path: Path):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(dir_path), num_labels=1
        ).to(self.device)

class HFRegressionModule(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HuggingFaceRegressionAdapter(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        freeze_encoder_parameters(self.model)
        print_trainable_parameters(self.model)

        #  Accelerate  DeepSpeed  ZeRO Stage 3
        deepspeed_plugin = None
        try:
            if DeepSpeedPlugin is not None:
                grad_accum = int(getattr(cfg, "grad_accum_steps", getattr(cfg, "gradient_accumulation_steps", 1)))
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=3,
                    gradient_accumulation_steps=grad_accum,
                    offload_optimizer_device="none",
                    offload_param_device="none",
                    gradient_clipping=getattr(cfg, "gradient_clipping", 1.0),
                )
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "1"
        except Exception as _:
            deepspeed_plugin = None

        mixed_precision = getattr(cfg, "mixed_precision", None)
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            mixed_precision=mixed_precision if mixed_precision else None,
        )
        self.device = self.accelerator.device
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.learning_rate)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def convert_numeric_examples(self, examples: List[Dict]):
        return self.model.convert_numeric_examples(examples)

    def compute_loss_and_metrics(self, batch: Dict):
        return self.model.compute_loss_and_metrics(batch)

    def fit(self, optimizer, train_loader, val_loader, num_epochs, checkpoint_dir, save_every_n_epochs):
        best_val = float("inf")
        best_dir = Path(checkpoint_dir) / "checkpoint_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / "model.pt"

        #
        periodic_dir = Path(checkpoint_dir) / "checkpoint_periodic"
        periodic_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            total_metrics = {}

            for batch in accelerate_tqdm(train_loader, desc=f"Train E{epoch+1}", disable=not self.accelerator.is_local_main_process):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, metrics = self.compute_loss_and_metrics(batch)
                # loss.backward()
                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

                #
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

            #
            train_loss = total_loss / max(1, len(train_loader))
            train_metrics = {f"train_{k}": v / max(1, len(train_loader)) for k, v in total_metrics.items()}

            #
            self.eval()
            val_loss_total = 0.0
            val_metrics_total = {}
            with torch.no_grad():
                for batch in accelerate_tqdm(val_loader, desc="Validating", disable=not self.accelerator.is_local_main_process):
                    loss, metrics = self.compute_loss_and_metrics(batch)
                    # 1.
                    # loss.unsqueeze(0)  [1]
                    gathered_loss = self.accelerator.gather_for_metrics(loss.unsqueeze(0))
                    # torch.sum
                    val_loss_total += torch.sum(gathered_loss).item()

                    # 2.
                    for key, value in metrics.items():
                        if key not in val_metrics_total:
                            val_metrics_total[key] = 0.0
                        #  Python
                        value_tensor = torch.tensor(value, device=self.device)
                        gathered_values = self.accelerator.gather_for_metrics(value_tensor)
                        val_metrics_total[key] += torch.sum(gathered_values).item()

            # 3.
            #  =  *
            num_total_val_batches = len(val_loader) * self.accelerator.num_processes

            #
            if num_total_val_batches > 0:
                val_loss = val_loss_total / num_total_val_batches
                val_metrics = {f"val_{k}": v / num_total_val_batches for k, v in val_metrics_total.items()}
            else:
                val_loss = 0.0
                val_metrics = {f"val_{k}": 0.0 for k in val_metrics_total.keys()}

            # ---  ( val_loss) ---
            #  val_loss
            if val_loss < best_val:
                best_val = val_loss
                #  is_main_process
                self.accelerator.save_state(output_dir=str(best_dir))
                if self.accelerator.is_main_process:
                    logger.info(f" {val_loss:.4f} {best_dir}")

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                periodic_path_dir = periodic_dir / f"epoch_{epoch+1}"
                #
                self.accelerator.save_state(output_dir=str(periodic_path_dir))
                if self.accelerator.is_main_process:
                    logger.info(f" {periodic_path_dir}")

            # ---  () ---
            if self.accelerator.is_main_process:
                #
                all_metrics_log = {**train_metrics, **val_metrics}
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in all_metrics_log.items())
                logger.info(f"Epoch {epoch+1}/{num_epochs} - {metrics_str}")

        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def test_dataset(self, test_loader, train_mean, train_std) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """test_rlm.py"""
        self.eval()
        #
        local_preds: List[float] = []
        local_targets: List[float] = []

        # test_rlm.py
        num_samples = 8  # test_rlm.py8
        temperature = 1.0
        top_p = 0.95

        for batch in accelerate_tqdm(test_loader, desc="Testing", disable=not self.accelerator.is_local_main_process):
            #
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            #
            allowed_input_keys = ["input_ids", "attention_mask"]
            inputs = {k: v for k, v in batch.items() if k in allowed_input_keys}

            #
            if not inputs:
                continue
            batch_size = next(iter(inputs.values())).size(0)

            # print(inputs)
            # assert 0

            # if "decoder_input_ids" not in inputs:
            #     decoder_start_token_id = self.model.config.decoder_start_token_id
            #     start_tokens = torch.full((labels_for_loss.size(0), 1), decoder_start_token_id, dtype=torch.long, device=labels_for_loss.device)
            #     shifted_labels = labels_for_loss[:, :-1]
            #     inputs["decoder_input_ids"] = torch.cat([start_tokens, shifted_labels], dim=-1)

            # 1. test_rlm.py
            gen_outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=self.model.n_out_tokens,
                min_new_tokens=self.model.n_out_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=temperature,
                num_return_sequences=num_samples,
            )

            # 2.
            if "y" in batch and isinstance(batch["y"], torch.Tensor):
                tgt_batch_vals_raw = batch["y"].detach().cpu().numpy().reshape(-1).tolist()
            else:
                tgt_batch_vals_raw = [0.0] * batch_size  #

            # 3. test_rlm.py
            for i in range(batch_size):
                #
                batch_predictions = []
                for sample_idx in range(num_samples):
                    seq_idx = i * num_samples + sample_idx
                    if seq_idx < len(gen_outputs):
                        try:
                            # tokenizertoken_ids_to_floats
                            decoded = self.model.tokenizer.token_ids_to_floats(gen_outputs[seq_idx].tolist())
                            if isinstance(decoded, list) and decoded:
                                pred_value = decoded[0]
                            else:
                                pred_value = float("nan")
                        except Exception:
                            pred_value = float("nan")
                        batch_predictions.append(pred_value)

                # test_rlm.py
                if batch_predictions:
                    valid_preds = [p for p in batch_predictions if not np.isnan(p)]
                    if valid_preds:
                        current_pred = float(np.median(valid_preds))
                    else:
                        current_pred = 0.0
                else:
                    current_pred = 0.0

                #
                current_target_raw = tgt_batch_vals_raw[i]

                #  (, )
                if current_target_raw is not None and np.isfinite(current_target_raw):
                    local_preds.append(current_pred)
                    local_targets.append(float(current_target_raw))

        #
        local_preds_tensor = torch.tensor(local_preds, dtype=torch.float32, device=self.accelerator.device)
        local_targets_tensor = torch.tensor(local_targets, dtype=torch.float32, device=self.accelerator.device)

        #
        all_predictions_tensor = self.accelerator.gather_for_metrics(local_preds_tensor)
        all_targets_tensor = self.accelerator.gather_for_metrics(local_targets_tensor)

        #
        if self.accelerator.is_main_process:
            pred_np = all_predictions_tensor.cpu().numpy()
            target_np = all_targets_tensor.cpu().numpy()

            #
            if len(pred_np) == 0:
                logger.error("/")
                return np.array([]), np.array([]), {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "rank_correlation": float("nan"), "r2": float("nan")}

            #  NaN  Inf
            valid_mask = np.isfinite(pred_np) & np.isfinite(target_np)
            if np.sum(valid_mask) == 0:
                 logger.error("")
                 return np.array([]), np.array([]), {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "rank_correlation": float("nan"), "r2": float("nan")}

            pred_np = pred_np[valid_mask]
            target_np = target_np[valid_mask]

            norm_pred, _, _ = normalize_data(pred_np, train_mean, train_std)
            norm_target, _, _ = normalize_data(target_np, train_mean, train_std)

            #
            mse = mean_squared_error(norm_target, norm_pred)
            mae = float(np.mean(np.abs(norm_target - norm_pred)))
            rank_corr, _ = spearmanr(norm_target, norm_pred)
            r2 = r2_score(norm_target, norm_pred)

            metrics = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": mae,
                "rank_correlation": float(rank_corr) if not np.isnan(rank_corr) else 0.0,
                "r2": float(r2),
            }

            logger.info(f" - MSE: {mse:.6f}, R2: {r2:.6f}, Rank Corr: {rank_corr:.6f}")
            return norm_pred, norm_target, metrics
        else:
            #
            return np.array([]), np.array([]), {}



def collate_fn(examples, model):
    #  convert_numeric_examples
    if hasattr(model, "convert_numeric_examples"):
        tensor_examples = model.convert_numeric_examples(examples)
        return tensor_examples
    return examples

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

def _resolve_checkpoint_path(path_str: str) -> Optional[Path]:
    """checkpointmodel.pt

    -  model.pt
    -  model.pt
    -  checkpoint_*  model.pt
    """
    if (
        path_str is None
        or str(path_str).strip() == ""
        or str(path_str).lower() == "none"
    ):
        return None
    p = Path(path_str)
    if not p.exists():
        logger.warning(f"init_checkpoint : {p}")
        return None
    #
    if p.is_file():
        return p
    #
    for fname in [
        "model.pt",
        "checkpoint.pt",
        # Hugging Face
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin.index.json",
    ]:
        direct_model = p / fname
        if direct_model.exists():
            return direct_model
    #  checkpoint_*
    # checkpoint_*.pt checkpoint_*/model.pt
    #
    file_candidates = sorted(
        p.glob("checkpoint_*.pt"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    if file_candidates:
        return file_candidates[0]
    #
    dir_candidates = [
        d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    if dir_candidates:
        dir_candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        latest = dir_candidates[0]
        for fname in [
            "model.pt",
            "checkpoint.pt",
            "pytorch_model.bin",
            "model.safetensors",
            "adapter_model.bin",
        ]:
            cand = latest / fname
            if cand.exists():
                return cand
        logger.warning(f" {latest}  model.pt  checkpoint.pt")
    else:
        logger.warning(f" {p}  checkpoint_*.pt  checkpoint_* ")
    return None


def _load_checkpoint_into_module(module: "HuggingFaceRegressionAdapter", ckpt_path: Path) -> bool:
    def try_load(state_dict_candidate):
        try:
            missing, unexpected = module.model.load_state_dict(
                state_dict_candidate, strict=False
            )
            return missing, unexpected, True
        except Exception:
            return None, None, False

    try:
        map_loc = module.model.device if hasattr(module.model, "device") else "cpu"
        data = torch.load(str(ckpt_path), map_location=map_loc)

        #  state_dict
        candidates = []
        if isinstance(data, dict):
            if "state_dict" in data and isinstance(data["state_dict"], dict):
                candidates.append(data["state_dict"])
            candidates.append(data)
        elif isinstance(data, (list, tuple)):
            #
            if len(data) > 0 and isinstance(data[0], dict):
                candidates.append(data[0])
        else:
            # state_dict
            candidates.append(data)

        #
        def with_prefix(state_dict, prefix):
            return {f"{prefix}{k}": v for k, v in state_dict.items()}

        def strip_prefix(state_dict, prefix):
            plen = len(prefix)
            return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}

        tried = []
        for state in candidates:
            if not isinstance(state, dict):
                continue
            # 1)
            tried.append(state)
            # 2)  module.
            tried.append(with_prefix(state, "module."))
            # 3)  module.
            tried.append(strip_prefix(state, "module."))
            # 4)  model.
            tried.append(strip_prefix(state, "model."))
            # 5)  state_dict.
            tried.append(strip_prefix(state, "state_dict."))
            # 6)  transformer.
            tried.append(strip_prefix(state, "transformer."))

        for cand in tried:
            if not cand:
                continue
            missing, unexpected, ok = try_load(cand)
            if ok:
                if missing:
                    logger.info(f"checkpoint: {len(missing)}")
                if unexpected:
                    logger.info(f"checkpoint: {unexpected}")
                logger.info(f" {ckpt_path} HF")
                return True

        logger.error("")
        return False
    except Exception as e:
        logger.error(f"checkpoint: {e}")
        return False

def load_train_stats(dataset_name, regression_data_dir):
    """


     JSONL  data_dir/train.jsonl  data_dir/dataset_name/train.jsonl
     'label''score''y' JSONL y_train.npy
    """
    train_y_path = os.path.join(regression_data_dir, dataset_name, "y_train.npy")
    try:
        # 1)  JSONL
        # data_dir/train.jsonl  data_dir/dataset_name/train.jsonl
        jsonl_candidates = [
            os.path.join(regression_data_dir, "train.jsonl"),
            os.path.join(regression_data_dir, dataset_name, "train.jsonl"),
        ]

        jsonl_path = None
        for cand in jsonl_candidates:
            if os.path.exists(cand):
                jsonl_path = cand
                break

        if jsonl_path is not None:
            count = 0
            mean_val = 0.0
            m2 = 0.0
            label_keys = ["label", "score", "y"]
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    value = None
                    for k in label_keys:
                        if k in obj:
                            try:
                                value = float(obj[k])
                            except Exception:
                                value = None
                            break
                    if value is None:
                        continue
                    count += 1
                    delta = value - mean_val
                    mean_val += delta / count
                    m2 += delta * (value - mean_val)

            if count > 0:
                var = m2 / count  #  numpy  ddof=0
                std_val = float(np.sqrt(var))
                if std_val == 0:
                    std_val = 1e-8
                print(" JSONL  mean/std ")
                return float(mean_val), std_val
            else:
                print(f":  {jsonl_path} ('label'/'score'/'y')")

        # 2)  NPY
        if os.path.exists(train_y_path):
            train_y = np.load(train_y_path)
            mean_val = np.mean(train_y)
            std_val = np.std(train_y)
            if std_val == 0:
                std_val = 1e-8
            print(" NPY  mean/std ")
            return float(mean_val), float(std_val)

        #
        print(
            f":  {regression_data_dir}  JSONL  NPY "
        )
        return None, None
    except Exception as e:
        print(f"{dataset_name}: {str(e)}")
        return None, None

def normalize_data(data, mean_val=None, std_val=None):
    """


    """
    data_array = np.array(data)

    if mean_val is not None and std_val is not None:
        #
        normalized_data = (data_array - mean_val) / std_val
        print("")
        return normalized_data, mean_val, std_val
    else:
        #
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val == 0:
            std_val = 1e-8
        normalized_data = (data_array - mean_val) / std_val
        return normalized_data.tolist(), mean_val, std_val


def get_all_dataset_names(data_dir: str) -> List[str]:
    """regression_data"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f": {data_dir}")

    dataset_names = []
    for item in data_path.iterdir():
        if item.is_dir() and (item / "info.json").exists():
            #
            required_files = [
                "N_train.npy",
                "N_val.npy",
                "N_test.npy",
                "y_train.npy",
                "y_val.npy",
                "y_test.npy",
            ]
            if all((item / f).exists() for f in required_files):
                dataset_names.append(item.name)

    return sorted(dataset_names)

def load_train_stats(train_dataset):
    train_mean, train_std = train_dataset.get_y_mean_std()
    return train_mean, train_std

def train_and_test_single_task(
    cfg: DictConfig, task_name: str, results_dir: Path
) -> Dict:
    """space"""
    logger.info(f": {task_name}")

    #
    task_cfg = cfg.copy()
    task_cfg.experiment_name = f"{cfg.experiment_name}_{task_name}"
    task_cfg.save_dir = str(results_dir / task_name / "checkpoints")

    #
    os.makedirs(task_cfg.save_dir, exist_ok=True)

    # RLMjsonl src/data/dataset/rlm_dataset.py
    dataset_name = task_name  #  "apps"
    data_dir = getattr(cfg.dataset, "data_dir", "data/code_metric")
    max_items = getattr(cfg.dataset, "max_items", None)

    # RLMDataset  max_samples max_items max_samples
    train_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="train", max_samples=max_items)
    val_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="val", max_samples=max_items)
    test_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="test", max_samples=max_items)

    train_mean, train_std = load_train_stats(train_dataset)

    #
    module = HuggingFaceRegressionAdapter(task_cfg)
    custom_collate = lambda examples: collate_fn(examples, module.model)

    init_ckpt = None
    if hasattr(task_cfg, "init_checkpoint"):
        init_ckpt = task_cfg.init_checkpoint
    if init_ckpt:
        resolved = _resolve_checkpoint_path(init_ckpt)
        if resolved is not None:
            #  HF from_pretrained
            if (
                resolved.is_dir()
                and hasattr(task_cfg, "hf")
                and getattr(task_cfg.hf, "enabled", False)
                and hasattr(module.model, "load_from_pretrained_dir")
            ):
                try:
                    module.model.load_from_pretrained_dir(resolved)
                    logger.info(f" Hugging Face : {resolved}")
                except Exception as e:
                    logger.warning(f"HF state_dict : {e}")
                    _load_checkpoint_into_module(module, resolved)
            else:
                _load_checkpoint_into_module(module, resolved)

    #
    train_loader = DataLoader(
        train_dataset,
        batch_size=task_cfg.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=task_cfg.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate,
    )

    # =====  =====
    module = HFRegressionModule(task_cfg)

    prepared_module, prepared_optimizer, prepared_train_loader, prepared_val_loader, prepared_test_loader = module.accelerator.prepare(
        module, module.optimizer, train_loader, val_loader, test_loader
    )
    prepared_module.fit(
        optimizer=prepared_optimizer,
        train_loader=prepared_train_loader,
        val_loader=prepared_val_loader,
        num_epochs=task_cfg.num_epochs,
        checkpoint_dir=task_cfg.save_dir,
        save_every_n_epochs=task_cfg.save_every_n_epochs,
    )

    #
    try:
    # best_dir
        best_dir = Path(task_cfg.save_dir) / "checkpoint_best"
        # best_dir = Path(task_cfg.save_dir) / "checkpoint_periodic" / "epoch_1"
        if best_dir.exists() and any(best_dir.iterdir()): #
            logger.info(f": {best_dir}")

            #  accelerator
            prepared_module.accelerator.load_state(str(best_dir))

            logger.info(f"")
        else:
            logger.warning(f" {best_dir}")
    except Exception as e:
        logger.error(f": {e}", exc_info=True)

    # checkpoint
    #
    #
    seed_everything(cfg.seed)
    if prepared_module.accelerator.is_main_process:
        logger.info(f": {cfg.seed}")


    predictions, targets, metrics = prepared_module.test_dataset(prepared_test_loader, train_mean, train_std)

    #
    if prepared_module.accelerator.is_main_process:
        task_results = {
            "dataset_name": dataset_name,
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
        }

        results_file = results_dir / dataset_name / f"results_{cfg.seed}.json"
        with open(results_file, "w") as f:
            json.dump(task_results, f, indent=2)

        logger.info(
            f" {dataset_name}  - MSE: {metrics.get('mse', float('nan')):.6f}, R2: {metrics.get('r2', float('nan')):.6f}, Rank Corr: {metrics.get('rank_correlation', float('nan')):.6f}"
        )
        return task_results

    #
    return {}


@hydra.main(config_path="../conf", config_name="config_rlm_example", version_base=None)
def main(cfg: DictConfig):
    """KBSSCDSSAPPS"""
    if cfg.use_wandb:
        swanlab.init(project=cfg.project_name, name=cfg.experiment_name)

    #  RLMDataset  dataset_name
    task_name = getattr(cfg.dataset, "task", "apps")
    logger.info(f" {task_name} RLM jsonl ")

    #
    seed_everything(cfg.seed)

    #
    base_results_dir = Path(f"results_exp_rlm_DIST2")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    #
    all_results = {}

    logger.info(f"\n=== : {task_name} ===")

    try:
        #
        task_results_dir = base_results_dir / task_name
        task_results_dir.mkdir(parents=True, exist_ok=True)

        #
        result = train_and_test_single_task(cfg, task_name, base_results_dir)

        if result:  #
            all_results[task_name] = result
            logger.info(f" {task_name} ")
            logger.info(f"MSE: {result['metrics']['mse']:.6f}")
            logger.info(f"R2: {result['metrics']['r2']:.6f}")
            logger.info(f"Rank Correlation: {result['metrics']['rank_correlation']:.6f}")

    except Exception as e:
        logger.error(f" {task_name} : {e}", exc_info=True)
        all_results[task_name] = {"error": str(e)}

    #
    if all_results:
        summary_file = base_results_dir / "summary_results.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        #
        print("\n===  ===")
        print(" | MSE | R2 | Rank Correlation")
        print("-" * 50)
        for task_name, result in all_results.items():
            if "error" in result:
                print(f"{task_name} | ERROR: {result['error']}")
            else:
                metrics = result.get('metrics', {})
                print(f"{task_name} | {metrics.get('mse', 'N/A'):.6f} | {metrics.get('r2', 'N/A'):.6f} | {metrics.get('rank_correlation', 'N/A'):.6f}")

    logger.info(f"\n: {base_results_dir}")


if __name__ == "__main__":
    main()
