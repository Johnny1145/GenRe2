import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import partial

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
import torch.distributed as dist
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.checkpoint import CheckpointManager
from src.utils.number_token_loss import NumberTokenLoss
from src.utils.reinforce_loss import ReinforceLoss
from src.utils.Remax_rlm import Remax
from typing import Any
# from ..model.base_module import BaseModule
# from ..model.regress_lm import core
# from ..model.regress_lm.models.pytorch import model as torch_model_lib
# from ..model.regress_lm.tokenizers import P10Tokenizer
# from ..model.regress_lm.vocabs import DecoderVocab, SentencePieceVocab
from src.data.dataset.rlm_dataset import RLMDataset

# # Initialize vocabs - Assuming these are not needed for HF-only path
# encoder_vocab = SentencePieceVocab.from_t5()
# decoder_vocab = DecoderVocab(tokenizer=P10Tokenizer())

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


class HuggingFaceRegressionAdapter(torch.nn.Module):
    """ Hugging Face """

    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                " transformers  Hugging Face : " + str(e)
            )

        local_dir = getattr(cfg.hf, "local_dir", None)
        model_name_or_path = local_dir if local_dir else cfg.hf.model_name_or_path
        trust_remote_code = getattr(cfg.hf, "trust_remote_code", True)
        logger.info(f"Loading model from: {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        self.n_out_tokens = getattr(self.model.config, "num_tokens_per_obj", 8) * getattr(self.model.config, "max_num_objs", 1)
        if hasattr(self.model.config, "num_tokens_per_obj") and hasattr(self.model.config, "max_num_objs"):
            self.n_out_tokens = self.model.config.num_tokens_per_obj * self.model.config.max_num_objs

        self.to(self.device)

    @property
    def device(self):
        """Dynamically get the device of the underlying model."""
        return next(self.model.parameters()).device

    def forward(self, **inputs):
        # --- FIX 1: CORRECT FORWARD PASS ---
        # The forward pass for training should call the model's core forward method
        # to get logits, not generate(), which is for inference.
        return self.model(**inputs)

    def convert_numeric_examples(self, examples: List[Dict]):
        """CE labels"""
        batch_inputs = {}
        if not examples:
            return {}

        texts = []
        targets = []

        # --- FIX 3: CORRECT DATA HANDLING ---
        # The append calls must be inside the loop to process all examples in the batch.
        for ex in examples:
            if isinstance(ex, dict):
                text = ex.get('input', ex.get('x', ''))
                target = float(ex.get('target', ex.get('y', 0)))
            else: # Assuming object with attributes
                text = getattr(ex, 'input', getattr(ex, 'x', ''))
                target = float(getattr(ex, 'target', getattr(ex, 'y', 0)))

            texts.append(text)
            targets.append(target)

        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        batch_inputs.update(enc)

        batch_inputs["y"] = torch.tensor(targets, dtype=torch.float32)

        labels_list = []
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Ensure float_to_token_ids method exists on the tokenizer
        if not hasattr(self.tokenizer, 'float_to_token_ids'):
             raise AttributeError("The provided tokenizer must have a 'float_to_token_ids' method.")

        for t in targets:
            ids = self.tokenizer.float_to_token_ids(float(t))
            if len(ids) < self.n_out_tokens:
                ids = ids + [pad_id] * (self.n_out_tokens - len(ids))
            elif len(ids) > self.n_out_tokens:
                ids = ids[:self.n_out_tokens]
            labels_list.append(ids)
        batch_inputs["labels"] = torch.tensor(labels_list, dtype=torch.long)

        start_id = getattr(self.model.config, "decoder_start_token_id", self.tokenizer.bos_token_id)
        if start_id is None:
            start_id = pad_id

        decoder_input_ids_list = []
        for ids in labels_list:
            decoder_ids = [start_id] + ids[:-1]
            decoder_input_ids_list.append(decoder_ids)

        decoder_input_ids = torch.tensor(decoder_input_ids_list, dtype=torch.long)
        decoder_attention_mask = (decoder_input_ids != pad_id).long()

        batch_inputs["decoder_input_ids"] = decoder_input_ids
        batch_inputs["decoder_attention_mask"] = decoder_attention_mask
        # print(batch_inputs)
        # assert 0

        return batch_inputs

    def load_from_pretrained_dir(self, dir_path: Path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(dir_path)).to(self.device)

    def token_ids_to_floats(self, token_ids: list[int]) -> list[float]:
        return self.tokenizer.token_ids_to_floats(token_ids)

    @property
    def decode_len(self) -> int:
        return self.n_out_tokens

    @property
    def max_num_objs(self) -> int:
        return 1

    def sample_with_logprobs(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int,
        temperature: float = 1.0,
        return_logits: bool = False,
        return_entropy: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.eval()
        device = next(self.model.parameters()).device
        allowed_input_keys = ["input_ids", "attention_mask"]
        inputs = {k: v.to(device) for k, v in batch.items() if k in allowed_input_keys}
        batch_size = inputs["input_ids"].shape[0]

        # Expand inputs for sampling
        expanded_inputs = {
            k: v.repeat_interleave(num_samples, dim=0) for k, v in inputs.items()
        }

        # generated_sequences_ids = torch.zeros(
        #     (batch_size * num_samples, self.decode_len), dtype=torch.long, device=device
        # )
        step_log_probs = torch.zeros(
            (batch_size * num_samples, self.decode_len), dtype=torch.float32, device=device
        )

        all_step_logits = [] if return_logits else None

        #
        step_entropies = None
        if return_entropy:
            step_entropies = torch.zeros(
                (batch_size * num_samples, self.decode_len),
                dtype=torch.float32,
                device=device,
            )

        decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", self.tokenizer.bos_token_id)
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id or 0


        # print(decoder_start_token_id)
        # assert 0
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1), decoder_start_token_id, dtype=torch.long, device=device
        )

        for step_idx in range(self.decode_len):
            model_inputs = expanded_inputs.copy()
            model_inputs["decoder_input_ids"] = current_tgt_ids

            outputs = self.model(**model_inputs)
            logits = outputs.logits[:, -1, :]

            if return_logits:
                all_step_logits.append(logits.cpu())

            scaled_logits = logits / temperature
            log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
            probs = torch.exp(log_probs)
            # print(probs)
            # assert 0

            # H = -sum(p * log(p))
            if return_entropy:
                with torch.no_grad():
                    p_log_p = probs * log_probs
                    # 0*inf=nan
                    p_log_p = torch.nan_to_num(p_log_p, nan=0.0, posinf=0.0, neginf=0.0)
                    entropy = -torch.sum(p_log_p, dim=-1)  # (B*S,)
                    step_entropies[:, step_idx] = entropy

            token_ids = torch.multinomial(probs, num_samples=1)
            # generated_sequences_ids[:, step_idx] = token_ids.squeeze(-1)

            chosen_log_prob = torch.gather(log_probs, 1, token_ids).squeeze(1)
            step_log_probs[:, step_idx] = chosen_log_prob

            current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        final_decoded_ids = current_tgt_ids.view(batch_size, num_samples, self.decode_len+1)
        final_step_log_probs = step_log_probs.view(batch_size, num_samples, self.decode_len)

        final_step_logits = None
        if return_logits and all_step_logits:
            final_step_logits = torch.stack(all_step_logits, dim=1).to(device) # (B*S, L, V)
            final_step_logits = final_step_logits.view(batch_size, num_samples, self.decode_len, -1)

        #
        policy_entropy = None
        if return_entropy:
            with torch.no_grad():
                #  (B, S, L_decode)
                step_entropies_reshaped = step_entropies.reshape(batch_size, num_samples, self.decode_len)
                #  (B, S)
                sample_entropies = step_entropies_reshaped.mean(dim=-1)
                #  (B)
                batch_entropies = sample_entropies.mean(dim=-1)
                #  ()
                policy_entropy = batch_entropies.mean()

        output_floats = np.zeros((batch_size, num_samples, self.max_num_objs), dtype=float)
        for b in range(batch_size):
            for s_idx in range(num_samples):
                # print(final_decoded_ids[b, s_idx, :].tolist())
                try:
                    decoded = self.token_ids_to_floats(final_decoded_ids[b, s_idx, :].tolist())
                except Exception:
                    decoded = None
                if isinstance(decoded, list) and decoded:
                    output_floats[b, s_idx, 0] = decoded[0]
                else:
                    output_floats[b, s_idx, 0] = np.nan

        return final_decoded_ids, output_floats, final_step_log_probs, final_step_logits, policy_entropy

    @torch.no_grad()
    def greedy_decode(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy Decoding

        Args:
            batch (Dict[str, torch.Tensor]):  'input_ids'  'attention_mask'
                                             'y'

        Returns:
            Tuple[np.ndarray, np.ndarray]:
        """
        self.eval()
        device = self.device
        allowed_input_keys = ["input_ids", "attention_mask"]
        inputs = {k: v.to(device) for k, v in batch.items() if k in allowed_input_keys}

        if not inputs:
            return np.array([]), np.array([])

        #  Hugging Face  generate
        # do_sample=False  num_beams=1
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.n_out_tokens,
            min_new_tokens=self.n_out_tokens,
            do_sample=False,
            num_beams=1,
        )

        # print(generated_ids.shape)
        # assert 0

        predictions = []
        for i in range(generated_ids.shape[0]):
            try:
                #  token ID
                # print(generated_ids[i].tolist())
                decoded_floats = self.token_ids_to_floats(generated_ids[i].tolist())
                # print(decoded_floats)
                #
                pred_value = decoded_floats[0] if decoded_floats else np.nan
            except Exception:
                pred_value = np.nan
            predictions.append(pred_value)

        return np.array(predictions)

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

class HFRegressionModule(torch.nn.Module):
    def __init__(self, cfg: DictConfig, y_quantile_transformer: Optional[Any] = None):
        super().__init__()
        self.cfg = cfg
        self.model = HuggingFaceRegressionAdapter(cfg)
        freeze_encoder_parameters(self.model)
        print_trainable_parameters(self.model)
        # self.ref_model = HuggingFaceRegressionAdapter(cfg)
        # self.ref_model.load_state_dict(self.model.state_dict())
        # for p in self.ref_model.parameters():
        #     p.requires_grad = False
        self.ref_model = None

        # --- FIX 5: CENTRALIZE LOSS INSTANTIATION ---
        self.remax = Remax(
            temperature=float(getattr(cfg, "temperature", 1.0)),
            num_samples=int(getattr(cfg, "num_samples", 4)),
            y_quantile_transformer=y_quantile_transformer,
        )

        deepspeed_plugin = None
        if getattr(cfg, "use_deepspeed", False) and DeepSpeedPlugin is not None:
            grad_accum = int(getattr(cfg, "grad_accum_steps", 1))
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=3,
                gradient_accumulation_steps=grad_accum,
                offload_optimizer_device="none",
                offload_param_device="none",
                gradient_clipping=getattr(cfg, "gradient_clipping", 1.0),
            )

        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            mixed_precision=getattr(cfg, "mixed_precision", "no"),
            gradient_accumulation_steps=int(getattr(cfg, "grad_accum_steps", 1)),
        )
        self.device = self.accelerator.device
        self.model.to(self.device)
        # self.ref_model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def convert_numeric_examples(self, examples: List[Dict]):
        return self.model.convert_numeric_examples(examples)

    # --- FIX 5: LOSS COMPUTATION MOVED HERE ---
    def compute_loss_and_metrics(self, batch: Dict):
        """
        Computes the GRPO loss using the main model and the reference model.
        """
        # The GRPO loss function is called here, where both models are accessible.
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss, metrics = self.remax(
            model=self.model,
            tokenizer=self.model.tokenizer,
            batch=batch,
            ref_model=self.ref_model
        )
        return total_loss, metrics

    def fit(self, optimizer, train_loader, val_loader, num_epochs, checkpoint_dir, save_every_n_epochs):
        best_val_reward = float("-inf")  # rewardreward
        best_dir = Path(checkpoint_dir) / "checkpoint_best"
        periodic_dir = Path(checkpoint_dir) / "checkpoint_periodic"

        # stepswanlab
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_metrics = {}

            pbar = accelerate_tqdm(train_loader, desc=f"Train E{epoch+1}", disable=not self.accelerator.is_local_main_process)
            for batch_idx, batch in enumerate(pbar):
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.compute_loss_and_metrics(batch)
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                # stepbatch
                if self.accelerator.is_main_process:
                    loss_value = loss.detach().float().item()
                    total_loss += loss_value

                    # step
                    step_metrics = {
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                        "train/loss": loss_value,
                    }

                    #
                    for key, value in metrics.items():
                        value_item = value.item() if torch.is_tensor(value) else value
                        step_metrics[f"train/{key}"] = value_item
                        total_metrics[key] = total_metrics.get(key, 0.0) + value_item

                    # swanlab
                    swanlab.log(step_metrics)

                    #
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

                    # step
                    global_step += 1

            # epochepoch
            if self.accelerator.is_main_process:
                num_batches = len(train_loader)
                train_loss_avg = total_loss / num_batches
                train_metrics_avg = {f"train/epoch_avg/{k}": v / num_batches for k, v in total_metrics.items()}
                logger.info(f"Epoch {epoch+1} Train Loss (avg): {train_loss_avg:.4f}")
                # epoch
                swanlab.log({
                    "train/epoch": epoch + 1,
                    "train/epoch_avg/loss": train_loss_avg,
                    **train_metrics_avg
                })

            # Validation
            self.model.eval()
            val_loss_total = 0.0
            val_reward_total = 0.0  # mean_reward_sampled
            val_metrics_total = {}
            val_step = 0
            with torch.no_grad():
                for batch in accelerate_tqdm(val_loader, desc="Validating", disable=not self.accelerator.is_local_main_process):
                    loss, metrics = self.compute_loss_and_metrics(batch)

                    # step
                    if self.accelerator.is_main_process:
                        loss_value = loss.detach().float().item()
                        val_loss_total += loss_value

                        # mean_reward_sampled
                        if "mean_reward_sampled" in metrics:
                            reward_value = metrics["mean_reward_sampled"]
                            reward_item = reward_value.item() if torch.is_tensor(reward_value) else reward_value
                            val_reward_total += reward_item

                        step_val_metrics = {
                            "val/step": global_step,  # stepx
                            "val/epoch": epoch + 1,
                            "val/loss": loss_value,
                        }

                        #
                        for key, value in metrics.items():
                            value_item = value.item() if torch.is_tensor(value) else value
                            step_val_metrics[f"val/{key}"] = value_item
                            val_metrics_total[key] = val_metrics_total.get(key, 0.0) + value_item

                        swanlab.log(step_val_metrics)
                        val_step += 1

                # 1.
            val_reward_avg = float("-inf")
            val_loss = 0.0
            if self.accelerator.is_main_process:
                num_val_batches = len(val_loader)
                if num_val_batches > 0:
                    val_loss = val_loss_total / num_val_batches
                    val_reward_avg = val_reward_total / num_val_batches
                    logger.info(f"Epoch {epoch+1} Val Loss (avg): {val_loss:.4f}, Val Reward (avg): {val_reward_avg:.4f}")

                    # epochswanlab
                    val_metrics_avg = {f"val/epoch_avg/{k}": v / num_val_batches for k, v in val_metrics_total.items()}
                    swanlab.log({
                        "val/epoch": epoch + 1,
                        "val/epoch_avg/loss": val_loss,
                        "val/epoch_avg/mean_reward_sampled": val_reward_avg,
                        **val_metrics_avg
                    })

            # 2.
            #  (1.0 for save, 0.0 for not save)
            #
            save_decision = torch.tensor(0.0, device=self.device)
            if self.accelerator.is_main_process:
                if val_reward_avg > best_val_reward:
                    best_val_reward = val_reward_avg
                    save_decision.fill_(1.0)

            #  `accelerator.wait_for_everyone()`
            self.accelerator.wait_for_everyone()

            # `torch.distributed.broadcast`
            #  rank 0  `save_decision`  `save_decision`
            dist.broadcast(save_decision, src=0)

            # 3.
            if save_decision.item() == 1.0:
                #  `save_decision`  1.0 save_state
                self.accelerator.save_state(output_dir=str(best_dir))
                if self.accelerator.is_main_process:
                    #
                    logger.info(f"reward {best_val_reward:.4f} {best_dir}")

            # Periodic checkpoint
            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                periodic_path_dir = periodic_dir / f"epoch_{epoch+1}"
                #
                self.accelerator.save_state(output_dir=str(periodic_path_dir))
                if self.accelerator.is_main_process:
                    logger.info(f" {periodic_path_dir}")

            # --- FIX 6: ACCELERATE-COMPATIBLE REF MODEL UPDATE ---
            # if (epoch + 1) % 5 == 0:
            #     self.accelerator.wait_for_everyone()
            #     # Get the state dict from the unwrapped model
            #     unwrapped_model = self.accelerator.unwrap_model(self.model)
            #     state_dict = unwrapped_model.state_dict()
            #     # Load it into the reference model (which is not under accelerator control)
            #     self.ref_model.load_state_dict(state_dict)
            #     if self.accelerator.is_main_process:
            #         logger.info("Refreshed reference model with current model weights.")
        print("here")
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def test_dataset(self, test_loader, train_mean, train_std) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """test_rlm.py"""
        self.eval()
        #
        local_preds: List[float] = []
        local_targets: List[float] = []

        # test_rlm.py
        num_samples = 64  # test_rlm.py8
        temperature = 1.0
        top_p = 0.95
        print("here3")

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

            # print(gen_outputs.shape)
            # assert 0

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
                        current_pred = float("nan")
                else:
                    current_pred = float("nan")

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

def load_train_stats(train_dataset):
    train_mean, train_std = train_dataset.get_y_mean_std()
    return train_mean, train_std

# --- FIX 4: MODIFIED COLLATE FUNCTION ---
def collate_fn_with_stats(examples, model, train_mean, train_std):
    """Custom collate function that also adds training stats to the batch."""
    batch = model.convert_numeric_examples(examples)
    if batch: # Ensure batch is not empty
        batch['y_mean'] = torch.tensor(train_mean, dtype=torch.float32)
        batch['y_std'] = torch.tensor(train_std, dtype=torch.float32)
    return batch

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


def train_and_test_single_task(
    cfg: DictConfig, task_name: str, results_dir: Path, sub_task: Optional[str] = None
) -> Dict:
    """space

     CDSS:
      - task_name  "cdss"
      - sub_task / "Bash"
      -  <data_dir>/cdss/<sub_task>
      -
    """
    full_task_name = f"{task_name}_{sub_task}" if sub_task else task_name
    logger.info(f": {full_task_name}")

    task_cfg = cfg.copy()
    task_cfg.experiment_name = f"{cfg.experiment_name}_{full_task_name}"
    #  sub_task
    if sub_task:
        task_cfg.save_dir = str(results_dir / task_name / sub_task / f"checkpoints_{cfg.seed}")
    else:
        task_cfg.save_dir = str(results_dir / task_name / f"checkpoints_{cfg.seed}")
    os.makedirs(task_cfg.save_dir, exist_ok=True)

    # RLMjsonl src/data/dataset/rlm_dataset.py
    #  CDSS<data_dir>/cdss/<sub_task>
    if task_name == "cdss" and sub_task is not None:
        dataset_name = f"{task_name}/{sub_task}"
    elif task_name == "nas" and sub_task is not None:
        dataset_name = sub_task.lower()
    else:
        dataset_name = task_name  #  "apps""kbss"
    data_dir = getattr(cfg.dataset, "data_dir", "data/code_metric")
    max_items = getattr(cfg.dataset, "max_items", None)

    train_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="train", max_samples=max_items)
    val_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="val", max_samples=max_items)
    test_dataset = RLMDataset(data_dir=data_dir, dataset_name=dataset_name, split="test", max_samples=max_items)

    train_mean, train_std = train_dataset.get_y_mean_std()
    y_quantile_transformer = train_dataset.get_y_quantile_transformer(random_state=cfg.seed)
    module = HFRegressionModule(task_cfg, y_quantile_transformer=y_quantile_transformer)

    # --- FIX 4: PASSING STATS TO DATALOADER ---
    custom_collate = partial(collate_fn_with_stats, model=module, train_mean=train_mean, train_std=train_std)

    train_loader = DataLoader(train_dataset, batch_size=task_cfg.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=task_cfg.batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)

    (
        prepared_module, prepared_optimizer,
        prepared_train_loader, prepared_val_loader, prepared_test_loader
    ) = module.accelerator.prepare(
        module, module.optimizer, train_loader, val_loader, test_loader
    )

    prepared_module.fit(
        optimizer=prepared_optimizer, train_loader=prepared_train_loader, val_loader=prepared_val_loader,
        num_epochs=task_cfg.num_epochs, checkpoint_dir=task_cfg.save_dir, save_every_n_epochs=task_cfg.save_every_n_epochs,
    )

    #
    try:
        # best_dir
        best_dir = Path(task_cfg.save_dir) / "checkpoint_best"
        if best_dir.exists() and any(best_dir.iterdir()):  #
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

    if prepared_module.accelerator.is_main_process:
        task_results = {
            "dataset_name": dataset_name, "metrics": metrics,
            "predictions": predictions.tolist(), "targets": targets.tolist(),
        }

        #  cdss/sub_task
        results_task_dir = results_dir / task_name / dataset_name if task_name == "nas" else results_dir / dataset_name
        results_task_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_task_dir / f"results_{cfg.seed}.json"
        with open(results_file, "w") as f:
            json.dump(task_results, f, indent=2)
        logger.info(f" {dataset_name}  - MSE: {metrics.get('mse', float('nan')):.6f}, R2: {metrics.get('r2', float('nan')):.6f}, Rank Corr: {metrics.get('rank_correlation', float('nan')):.6f}")
        return task_results

    return {}


@hydra.main(config_path="../conf", config_name="config_rlm_example", version_base=None)
def main(cfg: DictConfig):
    #
    # RANKaccelerate/torchrunLOCAL_RANK
    #
    rank = os.environ.get("RANK", None)
    local_rank = os.environ.get("LOCAL_RANK", None)

    if rank is not None:
        is_main_process = int(rank) == 0
    elif local_rank is not None:
        is_main_process = int(local_rank) == 0
    else:
        #
        is_main_process = True

    if is_main_process:
        swanlab_api_key = os.environ.get("SWANLAB_API_KEY")
        if swanlab_api_key:
            swanlab.login(api_key=swanlab_api_key)
        if cfg.use_wandb:
            swanlab.init(project=cfg.project_name, name="rlm_rl", config=dict(cfg), tags=[cfg.dataset.task])

    #  RLMDataset  dataset_name
    task_name = getattr(cfg.dataset, "task", "apps")
    sub_task = getattr(cfg.dataset, "sub_task", None)
    if task_name == "cdss" and sub_task is not None:
        display_task_name = f"{task_name}_{sub_task}"
    else:
        display_task_name = task_name
    logger.info(f" {display_task_name} RLM jsonl ")

    seed_everything(cfg.seed)

    base_results_dir = Path("results_exp_rlm/results_exp_rlm_remax_after_ce")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    #
    all_results = {}

    logger.info(f"\n=== : {display_task_name} ===")

    try:
        # cdss  sub_task
        if task_name == "cdss" and sub_task is not None:
            task_results_dir = base_results_dir / task_name / sub_task
        else:
            task_results_dir = base_results_dir / task_name
        task_results_dir.mkdir(parents=True, exist_ok=True)

        #
        result = train_and_test_single_task(cfg, task_name, base_results_dir, sub_task=sub_task)

        if result:  #
            key_name = display_task_name
            all_results[key_name] = result
            logger.info(f" {display_task_name} ")
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

    logger.info(f"\nAll tasks finished! Results are in: {base_results_dir}")
    if is_main_process:
        swanlab.finish()

if __name__ == "__main__":
    main()
