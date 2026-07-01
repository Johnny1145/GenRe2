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
from accelerate.utils import tqdm as accelerate_tqdm
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.checkpoint import CheckpointManager
from src.utils.number_token_loss import NumberTokenLoss
from src.utils.reinforce_mse_reward_normalized import Remax_mse

from ..data.dataset.numeric_regression_binary_fit_dataset import Binary_fit_Dataset
from ..model.base_module import BaseModule
from ..model.regress_lm import core
from ..model.regress_lm.models.pytorch import mlp_encoder_RL_model_kl as torch_model_lib
from ..model.regress_lm.tokenizers import NormalizedTokenizer
from ..model.regress_lm.vocabs import DecoderVocab, SentencePieceVocab

# Initialize vocabs (decoder_vocab will be reconfigured from cfg later)
encoder_vocab = SentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=NormalizedTokenizer())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(examples, model, split: str = "train"):
    if split == "train":
        tensor_examples = model.convert_normalized_RL_examples(examples)
    else:
        # val or test
        tensor_examples = model.convert_normalized_RL_examples_test(examples)
    return tensor_examples

def configure_decoder_vocab_from_cfg(cfg: DictConfig):
    """ cfg.base/cfg.digits  decoder_vocab"""
    global decoder_vocab
    decoder_vocab = DecoderVocab(tokenizer=NormalizedTokenizer(num_digits=cfg.digits, base=cfg.base))

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
    for fname in ["model.pt", "checkpoint.pt"]:
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
        for fname in ["model.pt", "checkpoint.pt"]:
            cand = latest / fname
            if cand.exists():
                return cand
        logger.warning(f" {latest}  model.pt  checkpoint.pt")
    else:
        logger.warning(f" {p}  checkpoint_*.pt  checkpoint_* ")
    return None


def _load_checkpoint_into_module(module: "RegressionModule", ckpt_path: Path) -> bool:
    try:
        map_loc = module.model.device if hasattr(module.model, "device") else "cpu"
        data = torch.load(str(ckpt_path), map_location=map_loc)
        state = (
            data["state_dict"]
            if isinstance(data, dict) and "state_dict" in data
            else data
        )
        missing, unexpected = module.model.load_state_dict(state, strict=False)
        if missing:
            # state
            new_state = {}
            for key, value in state.items():
                new_state[f"module.{key}"] = value
            missing, unexpected = module.model.load_state_dict(new_state, strict=False)
        if unexpected:
            logger.info(f"checkpoint: {unexpected}")
        logger.info(f" {ckpt_path} RL...")
        return True
    except Exception as e:
        logger.error(f"checkpoint: {e}")
        return False


class RegressionModule(BaseModule):
    def __init__(self, cfg: DictConfig):
        # MLP encoder
        mlp_kwargs = {}
        if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type == "mlp":
            # MLP encoder
            if hasattr(cfg.model, "input_dim"):
                mlp_kwargs["input_dim"] = cfg.model.input_dim
            if hasattr(cfg.model, "hidden_dims"):
                # ListConfig
                hidden_dims_list = OmegaConf.to_container(cfg.model.hidden_dims, resolve=True)
                if not isinstance(hidden_dims_list, list):
                    hidden_dims_list = [hidden_dims_list]
                mlp_kwargs["hidden_dims"] = hidden_dims_list * cfg.model.num_decoder_layers
            if hasattr(cfg.model, "output_dim"):
                mlp_kwargs["output_dim"] = cfg.model.d_model

        model = torch_model_lib.PyTorchModel(
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            max_input_len=cfg.model.max_input_len,
            max_num_objs=cfg.model.max_num_objs,
            d_model=cfg.model.d_model,
            num_encoder_layers=cfg.model.num_decoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            encoder_type=cfg.model.encoder_type,
            **mlp_kwargs,  # MLP encoder
        )

        criterion = None

        super().__init__(
            model=model,
            criterion=criterion,
            project_name=cfg.project_name,
            experiment_name=cfg.experiment_name,
            use_wandb=cfg.use_wandb,
            log_dir=cfg.log_dir,
        )
        self.cfg = cfg
        self.total_steps = None
        self.best_save_metric = cfg.get("best_save_metric", "val_loss")
        self.reinforce_loss_fn = Remax_mse(
            temperature=cfg.reinforce.get("temperature", 1.0),
            num_samples=cfg.reinforce.get("num_samples", 8),
            reward_scale=cfg.reinforce.get("reward_scale", 1.0),
            # kl_weight=cfg.reinforce.get("kl_weight", 0),
            kl_weight=0,
        )
        self.best_val_mean_reward = float("-inf")
        self.ref_model = None  # KL
        self.save = None
        #
        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "train_reward": [],
            "train_mse": [],
            "train_r2": [],
            "val_loss": [],
            "val_reward": [],
            "val_mse": [],
            "val_r2": [],
            "kl_loss": [],
            "learning_rate": [],
            "train_entropy": [],
            "val_entropy": [],
            "test_mse": [],
            "test_r2": [],
        }

    def set_total_steps(self, total_steps: int):
        """Set total training steps for scheduler configuration"""
        self.total_steps = total_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate
        )

        if not hasattr(self.cfg, "scheduler") or self.total_steps is None:
            return {"optimizer": optimizer}

        # Create scheduler based on configuration
        scheduler_config = self.cfg.scheduler
        scheduler_type = scheduler_config.get("type", "constant")

        if scheduler_type == "cosine":
            # Calculate minimum learning rate
            min_lr = self.cfg.learning_rate * scheduler_config.get("min_lr_ratio", 0.1)

            # Create cosine annealing scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - scheduler_config.get("warmup_steps", 0),
                eta_min=min_lr,
            )

            # Add warmup if specified
            warmup_steps = scheduler_config.get("warmup_steps", 0)
            if warmup_steps > 0:

                def warmup_lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        return 1.0

                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=warmup_lr_lambda
                )

                # Use sequential scheduler to combine warmup and cosine
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                )

        elif scheduler_type == "linear":
            # Linear decay scheduler
            def linear_lr_lambda(step):
                if step < scheduler_config.get("warmup_steps", 0):
                    return step / scheduler_config.get("warmup_steps", 0)
                else:
                    remaining_steps = self.total_steps - step
                    total_decay_steps = self.total_steps - scheduler_config.get(
                        "warmup_steps", 0
                    )
                    return max(
                        scheduler_config.get("min_lr_ratio", 0.1),
                        remaining_steps / total_decay_steps,
                    )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=linear_lr_lambda
            )

        else:  # constant
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        return {"optimizer": optimizer, "scheduler": scheduler}

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_reward = 0
        total_kl_loss = 0
        num_batches = 0
        total_entropy = 0
        if self.cfg.if_ntl:
            self.NTL = NumberTokenLoss(self.model.decoder_vocab, self.model.device)
        else:
            self.NTL = None

        progress_bar = accelerate_tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                # print(batch)
                # assert 0

                loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
                    batch,
                    self.NTL,
                    self.reinforce_loss_fn,
                    ref_model=self.ref_model,
                    )

                self.accelerator.backward(loss)

                # =================================================================

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_reward += metrics["mean_reward"]
                total_entropy += metrics["policy_entropy"]  #
                if "kl_loss" in metrics:
                    total_kl_loss += metrics["kl_loss"]
                num_batches += 1
                self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        avg_reward = total_reward / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader) if num_batches > 0 else 0.0

        return {
            "train_loss": avg_loss,
            "train_mean_reward": avg_reward,
            "train_kl_loss": avg_kl_loss,
            "train_entropy": total_entropy / len(self.train_loader),  #
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        if not hasattr(self, "val_loader"):
            return {}

        self.model.eval()
        total_loss = 0
        total_reward = 0
        total_kl_loss = 0
        num_batches = 0
        total_entropy = 0
        progress_bar = accelerate_tqdm(
            self.val_loader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
                batch, self.NTL, self.reinforce_loss_fn, ref_model=self.ref_model
            )
            total_loss += loss.item()
            total_reward += metrics["mean_reward"]
            total_entropy += metrics["policy_entropy"]

            if "kl_loss" in metrics:
                total_kl_loss += metrics["kl_loss"]
            num_batches += 1

        val_loss = total_loss / len(self.val_loader)
        val_mean_reward = total_reward / len(self.val_loader)
        val_kl_loss = total_kl_loss / len(self.val_loader) if num_batches > 0 else 0.0
        val_entropy = total_entropy / len(self.val_loader) if num_batches > 0 else 0.0  #

        if self.best_save_metric == "val_loss":
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                if self.cfg.save_dir and self.accelerator.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        self, "best_ce", {"val_loss": val_loss}
                    )
        elif self.best_save_metric == "val_mean_reward":
            if val_mean_reward > self.best_val_mean_reward:
                self.best_val_mean_reward = val_mean_reward
                self.early_stop_counter = 0
                if self.cfg.save_dir and self.accelerator.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        self, "best", {"val_mean_reward": val_mean_reward}
                    )

        return {
            "val_loss": val_loss,
            "val_mean_reward": val_mean_reward,
            "val_kl_loss": val_kl_loss,
            "val_entropy": val_entropy,  #
        }

    # @torch.no_grad()
    # def evaluate_loader(self, loader) -> Dict[str, float]:
    #     """loaderloss/mean_reward/median_reward/kl/entropy"""
    #     self.model.eval()
    #     total_loss = 0.0
    #     total_mean_reward = 0.0
    #     total_median_reward = 0.0
    #     total_kl = 0.0
    #     total_entropy = 0.0
    #     num_batches = 0

    #     for batch in loader:
    #         batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
    #         loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
    #             batch, self.NTL, self.reinforce_loss_fn, ref_model=self.ref_model
    #         )
    #         total_loss += float(loss.item())
    #         total_mean_reward += float(metrics.get("mean_reward", 0.0))
    #         total_median_reward += float(metrics.get("median_reward", 0.0))
    #         total_entropy += float(metrics.get("policy_entropy", 0.0))
    #         if "kl_loss" in metrics:
    #             total_kl += float(metrics["kl_loss"])
    #         num_batches += 1

    #     denom = max(1, num_batches)
    #     return {
    #         "loss": total_loss / denom,
    #         "mean_reward": total_mean_reward / denom,
    #         "median_reward": total_median_reward / denom,
    #         "kl_loss": total_kl / denom,
    #         "entropy": total_entropy / denom,
    #     }

    def fit(
        self, train_loader, val_loader, num_epochs, checkpoint_dir, save_every_n_epochs
    ):
        # fit
        #
        if not hasattr(self, "train_loader"):
            self.prepare(train_loader, val_loader)
            # ref_model
            if self.ref_model is not None and hasattr(self.model, 'device'):
                self.ref_model = self.ref_model.to(self.model.device)
                logger.info(f"ref_model: {self.model.device}")

        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        if self.logger:
            param_info = self.get_model_parameters()
            self.logger.log(param_info)
            self.accelerator.print(f"Model parameters logged: {param_info}")

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.epoch_counter = 0
        self.early_stop_patience = 20

        if self.cfg.if_ntl:
            self.NTL = NumberTokenLoss(self.model.decoder_vocab, self.model.device)
        else:
            self.NTL = None

        # initial_train_metrics = self.test_dataset_normalized(self.train_loader)[6]
        # initial_val_metrics = self.test_dataset_normalized(self.val_loader)[6]
        # initial_test_metrics = self.test_dataset_normalized(self.test_loader)[6]
        # initial_train_metrics = {**initial_train_metrics, **self.evaluate_loader(self.train_loader)}
        # initial_val_metrics = {**initial_val_metrics, **self.evaluate_loader(self.val_loader)}
        # #  test_dataset_normalized  metrics_median ( 6)
        # current_lr = self.optimizer.param_groups[0]['lr']
        # self.training_history["epoch"].append(0)
        # self.training_history["train_loss"].append(float(initial_train_metrics.get("loss", 0.0)))
        # self.training_history["train_reward"].append(float(initial_train_metrics.get("mean_reward", 0.0)))
        # self.training_history["train_mse"].append(float(initial_train_metrics.get("mse", 0.0)))
        # self.training_history["train_r2"].append(float(initial_train_metrics.get("r2", 0.0)))
        # self.training_history["val_loss"].append(float(initial_val_metrics.get("loss", 0.0)))
        # self.training_history["val_reward"].append(float(initial_val_metrics.get("mean_reward", 0.0)))
        # self.training_history["kl_loss"].append(float(initial_train_metrics.get("kl_loss", 0.0)))
        # self.training_history["learning_rate"].append(float(current_lr))
        # self.training_history["train_entropy"].append(float(initial_train_metrics.get("entropy", 0.0)))
        # self.training_history["val_entropy"].append(float(initial_val_metrics.get("entropy", 0.0)))
        # self.training_history["test_mse"].append(float(initial_test_metrics.get("mse", 0.0)))
        # self.training_history["test_r2"].append(float(initial_test_metrics.get("r2", 0.0)))
        # self.training_history["val_mse"].append(float(initial_val_metrics.get("mse", 0.0)))
        # self.training_history["val_r2"].append(float(initial_val_metrics.get("r2", 0.0)))
        # baseline_metrics = {
        #     "train_loss": self.training_history["train_loss"][-1],
        #     "train_mean_reward": self.training_history["train_reward"][-1],
        #     "train_mse": self.training_history["train_mse"][-1],
        #     "train_r2": self.training_history["train_r2"][-1],
        #     "val_loss": self.training_history["val_loss"][-1],
        #     "val_mean_reward": self.training_history["val_reward"][-1],
        #     "val_mse": self.training_history["val_mse"][-1],
        #     "val_r2": self.training_history["val_r2"][-1],
        #     "train_kl_loss": self.training_history["kl_loss"][-1],
        #     "train_entropy": self.training_history["train_entropy"][-1],
        #     "val_entropy": self.training_history["val_entropy"][-1],
        #     "test_mse": self.training_history["test_mse"][-1],
        #     "test_r2": self.training_history["test_r2"][-1],
        # }
        # if self.logger:
        #     self.logger.log({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        # metrics_str = ", ".join(f"baseline_{k}: {v:.4f}" for k, v in baseline_metrics.items())
        # self.accelerator.print(f"Baseline (epoch 0): {metrics_str}")

        for epoch in range(num_epochs):
            self.epoch_counter = epoch
            # train_metrics = {"train_"+ k : v for k,v in self.test_dataset(train_loader)[2].items()}
            # train_metrics = {**self.train_epoch(),**train_metrics}
            # val_metrics = {"val_"+ k : v for k,v in self.test_dataset(val_loader)[2].items()}
            # val_metrics = {**self.validate_epoch(),**val_metrics}
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            metrics = {**train_metrics, **val_metrics}

            #
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history["epoch"].append(epoch + 1)
            self.training_history["train_loss"].append(float(train_metrics.get("train_loss", 0.0)))
            self.training_history["train_reward"].append(float(train_metrics.get("train_mean_reward", 0.0)))
            self.training_history["val_loss"].append(float(val_metrics.get("val_loss", 0.0)))
            self.training_history["val_reward"].append(float(val_metrics.get("val_mean_reward", 0.0)))
            self.training_history["kl_loss"].append(float(train_metrics.get("train_kl_loss", 0.0)))
            self.training_history["learning_rate"].append(float(current_lr))
            self.training_history["train_entropy"].append(float(train_metrics.get("train_entropy", 0.0)))  #
            self.training_history["val_entropy"].append(float(val_metrics.get("val_entropy", 0.0)))  #
            self.training_history["train_mse"].append(float(train_metrics.get("train_mse", 0.0)))  # MSE
            self.training_history["train_r2"].append(float(train_metrics.get("train_r2", 0.0)))  # R2
            self.training_history["val_mse"].append(float(val_metrics.get("val_mse", 0.0)))  # MSE
            self.training_history["val_r2"].append(float(val_metrics.get("val_r2", 0.0)))  # R2

            if self.logger:
                self.logger.log(metrics)

            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(
                f"Epoch [{epoch+1}/{num_epochs}]: {metrics_str}"
            )

            # JSON
            if self.accelerator.is_main_process and checkpoint_dir:
                history_file = Path(checkpoint_dir) / "training_history.json"
                with open(history_file, "w") as f:
                    json.dump(self.training_history, f, indent=2)

            if self.accelerator.is_main_process and checkpoint_dir:
                if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                    self.checkpoint_manager.save_checkpoint(self, epoch + 1, metrics)

            if self.early_stop_counter >= self.early_stop_patience:
                self.accelerator.print("Early stopping triggered")
                if self.accelerator.is_main_process and checkpoint_dir:
                    self.checkpoint_manager.save_checkpoint(self, epoch, metrics)
                break

        if self.accelerator.is_main_process and checkpoint_dir:
            self.checkpoint_manager.save_checkpoint(self, num_epochs, metrics)
            #
            history_file = Path(checkpoint_dir) / "training_history.json"
            with open(history_file, "w") as f:
                json.dump(self.training_history, f, indent=2)

        self.accelerator.end_training()

    def save_training_history(self, checkpoint_dir: str):
        """JSON"""
        if self.accelerator.is_main_process and checkpoint_dir:
            history_file = Path(checkpoint_dir) / "training_history.json"
            with open(history_file, "w") as f:
                json.dump(self.training_history, f, indent=2)
            self.accelerator.print(f": {history_file}")

    @staticmethod
    def load_training_history(checkpoint_dir: str) -> Dict:
        """JSON"""
        history_file = Path(checkpoint_dir) / "training_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                return json.load(f)
        else:
            print(f": {history_file}")
            return {}


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

def load_best_params_from_optuna(dataset_name: str, optuna_results_dir: str = "results_optuna_ce") -> Optional[Dict]:
    """results_optuna_cebest_params.json"""
    # {dataset_name}/{dataset_name}/best_params.json  {dataset_name}/best_params.json
    optuna_dir = Path(optuna_results_dir)

    # 1: {dataset_name}/{dataset_name}/best_params.json
    params_path = optuna_dir / dataset_name / dataset_name / "best_params.json"
    if params_path.exists():
        logger.info(f" {params_path} ")
        with open(params_path, "r") as f:
            return json.load(f)

    # 2: {dataset_name}/best_params.json
    params_path = optuna_dir / dataset_name / "best_params.json"
    if params_path.exists():
        logger.info(f" {params_path} ")
        with open(params_path, "r") as f:
            return json.load(f)

    logger.warning(f" {dataset_name} ")
    return None


def update_cfg_with_best_params(cfg: DictConfig, best_params: Dict) -> DictConfig:
    """best_params"""
    #
    if "learning_rate" in best_params:
        cfg.learning_rate = best_params["learning_rate"]
        logger.info(f" learning_rate: {cfg.learning_rate}")

    # basedigitsdecoder_vocab
    if "base" in best_params:
        cfg.base = best_params["base"]
        logger.info(f" base: {cfg.base}")

    if "digits" in best_params:
        cfg.digits = best_params["digits"]
        logger.info(f" digits: {cfg.digits}")

    #
    if "d_model" in best_params:
        cfg.model.d_model = best_params["d_model"]
        logger.info(f" d_model: {cfg.model.d_model}")

    if "nhead" in best_params:
        cfg.model.nhead = best_params["nhead"]
        logger.info(f" nhead: {cfg.model.nhead}")

    if "num_decoder_layers" in best_params:
        cfg.model.num_decoder_layers = best_params["num_decoder_layers"]
        logger.info(f" num_decoder_layers: {cfg.model.num_decoder_layers}")

    if "dim_feedforward" in best_params:
        cfg.model.dim_feedforward = best_params["dim_feedforward"]
        logger.info(f" dim_feedforward: {cfg.model.dim_feedforward}")

    # MLP encoderhidden_dim
    if "hidden_dim" in best_params:
        if hasattr(cfg.model, "hidden_dims"):
            # hidden_dims
            # hidden_dim
            #
            if isinstance(cfg.model.hidden_dims, list):
                #
                #
                cfg.model.hidden_dims = [best_params["hidden_dim"]]
            else:
                cfg.model.hidden_dims = [best_params["hidden_dim"]]
        else:
            # hidden_dims
            cfg.model.hidden_dims = [best_params["hidden_dim"]]
        logger.info(f" hidden_dims: {cfg.model.hidden_dims}")

    return cfg

def train_and_test_single_task(
    cfg: DictConfig, dataset_name: str, results_dir: Path
) -> Dict:
    """"""
    logger.info(f": {dataset_name}")

    #
    task_cfg = cfg.copy()
    task_cfg.dataset.params.data_dir = cfg.dataset.params.data_dir
    task_cfg.experiment_name = f"{cfg.experiment_name}_{dataset_name}"
    task_cfg.save_dir = str(results_dir / dataset_name / f"checkpoints_{cfg.seed}")

    #
    os.makedirs(task_cfg.save_dir, exist_ok=True)

    #
    train_dataset = Binary_fit_Dataset(
        data_dir=task_cfg.dataset.params.data_dir,
        split="train",
        dataset_name=dataset_name,
    )
    val_dataset = Binary_fit_Dataset(
        data_dir=task_cfg.dataset.params.data_dir,
        split="val",
        dataset_name=dataset_name,
    )
    test_dataset = Binary_fit_Dataset(
        data_dir=task_cfg.dataset.params.data_dir,
        split="test",
        dataset_name=dataset_name,
    )

    task_dim = train_dataset.dimension

    if hasattr(task_cfg.model, "encoder_type") and task_cfg.model.encoder_type == "mlp":
        task_cfg.model.input_dim = task_dim
        logger.info(f"MLP encoderinput_dim: {task_dim}")

    #
    module = RegressionModule(task_cfg)
    ref_module = RegressionModule(task_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=task_cfg.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, module.model, split="train"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=task_cfg.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, module.model, split="val"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=task_cfg.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, module.model, split="test"),
    )

    if cfg.eval_mode:
        module.prepare(train_loader, val_loader)
        ref_module.prepare(train_loader, val_loader)

    init_ckpt = (
        task_cfg.init_checkpoint
        if hasattr(task_cfg, "init_checkpoint") and task_cfg.init_checkpoint
        else f"results_search_mlp_encoder_ce/{dataset_name}/{dataset_name}/checkpoints_{cfg.seed}/checkpoint_best/model.pt"
    )
    # init_ckpt = f"results_search_mlp_encoder_remax_test/{dataset_name}/{dataset_name}/checkpoints_{cfg.seed}/checkpoint_1/model.pt"
    if init_ckpt:
        resolved = _resolve_checkpoint_path(init_ckpt)
        if resolved is not None:
            res = _load_checkpoint_into_module(module, resolved)
            ref_res = _load_checkpoint_into_module(ref_module, resolved)
            print(res,ref_res)
            if res and ref_res:
                logger.info(f" {init_ckpt} ")
                # KL
                module.ref_model = ref_module.model
                # ref_model
                if hasattr(module.model, 'device'):
                    module.ref_model = module.ref_model.to(module.model.device)
                logger.info("KL")
            else:
                assert False

    #
    #
    if not cfg.eval_mode:
        #
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * task_cfg.num_epochs
        module.set_total_steps(total_steps)

        logger.info(
            f" {dataset_name}: {steps_per_epoch} steps/epoch x {task_cfg.num_epochs} epochs = {total_steps} total steps"
        )
        # module.test_loader = test_loader
        #
        module.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=task_cfg.num_epochs,
            checkpoint_dir=task_cfg.save_dir,
            save_every_n_epochs=task_cfg.save_every_n_epochs,
        )

    #
    ckpt_path = task_cfg.save_dir + "/checkpoint_best/model.pt"
    # results_dir1 = Path(f"results_P10_mlp_encoder_pure_Reinforce_greedy_kl_big_batch/{dataset_name}")
    # ckpt_path = str(results_dir1 / dataset_name / "checkpoints") + "/checkpoint_best/model.pt"
    resolved = _resolve_checkpoint_path(ckpt_path)
    if resolved is not None:
        _load_checkpoint_into_module(module, resolved)
    seed_everything(cfg.seed)
    predictions_mean, predictions_median, predictions_clip_mean, predictions_clip_median, targets, metrics_mean, metrics_median, metrics_clip_mean, metrics_clip_median = module.test_dataset_normalized(test_loader)
    #
    task_results = {
        "dataset_name": dataset_name,
        "metrics_mean": metrics_mean,
        "metrics_median": metrics_median,
        "metrics_clip_mean": metrics_clip_mean,
        "metrics_clip_median": metrics_clip_median,
        "predictions_mean": predictions_mean.tolist(),
        "predictions_median": predictions_median.tolist(),
        "predictions_clip_mean": predictions_clip_mean.tolist(),
        "predictions_clip_median": predictions_clip_median.tolist(),
        "targets": targets.tolist(),
    }
    #

    results_file = results_dir / dataset_name / f"results_seed_{cfg.seed}.json"
    with open(results_file, "w") as f:
        json.dump(task_results, f, indent=2)

    return task_results


@hydra.main(
    config_path="../conf", config_name="config_multi_task_reinforce", version_base=None
)
def main(cfg: DictConfig):
    """"""
    if cfg.use_wandb:
        swanlab.init(project=cfg.project_name, name=cfg.experiment_name)

    #
    dataset_name = cfg.dataset.name
    logger.info(f" {dataset_name} RL")

    #  cfg  decoder_vocab Module/Model  tokenizer
    best_params = load_best_params_from_optuna(dataset_name)
    if best_params is not None:
        cfg = update_cfg_with_best_params(cfg, best_params)
        logger.info(f": {best_params}")
    # print(cfg)
    # assert 0

    #  cfg  decoder_vocab Module/Model  tokenizer
    configure_decoder_vocab_from_cfg(cfg)
    #
    seed_everything(cfg.seed)
    #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_search_mlp_encoder_remax/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # # results.json
    if cfg.skip_mode:
        results_json_path =Path(f"results_search_mlp_encoder_remax/{dataset_name}/{dataset_name}/results_seed_{cfg.seed}.json")
        if results_json_path.exists():
            logger.info(f"{dataset_name} results.json: {results_dir}")
            return

    #
    result = train_and_test_single_task(cfg, dataset_name, results_dir)

    logger.info(f" {dataset_name} : {results_dir}")
    logger.info(f"MSE_mean: {result['metrics_mean']['mse']:.6f}")
    logger.info(f"Rank Corr_mean: {result['metrics_mean']['rank_correlation']:.6f}")
    logger.info(f"MSE_median: {result['metrics_median']['mse']:.6f}")
    logger.info(f"Rank Corr_median: {result['metrics_median']['rank_correlation']:.6f}")
    logger.info(f"MSE_clip_mean: {result['metrics_clip_mean']['mse']:.6f}")
    logger.info(f"Rank Corr_clip_mean: {result['metrics_clip_mean']['rank_correlation']:.6f}")
    logger.info(f"MSE_clip_median: {result['metrics_clip_median']['mse']:.6f}")
    logger.info(f"Rank Corr_clip_median: {result['metrics_clip_median']['rank_correlation']:.6f}")


if __name__ == "__main__":
    main()
