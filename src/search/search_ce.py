import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
from optuna.trial import Trial
import pandas as pd
import rootutils
import swanlab
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
import random
from sqlalchemy.exc import OperationalError
import time
import gc

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.checkpoint import CheckpointManager
from src.utils.number_token_loss import NumberTokenLoss
from src.utils.reinforce_loss import ReinforceLoss

from ..data.dataset.numeric_regression_binary_fit_dataset import Binary_fit_Dataset
from ..model.base_module_new import BaseModule
from ..model.regress_lm import core
from ..model.regress_lm.models.pytorch import model as torch_model_lib
from ..model.regress_lm.tokenizers import NormalizedTokenizer
from ..model.regress_lm.vocabs import DecoderVocab, SentencePieceVocab

# Initialize vocabs
encoder_vocab = SentencePieceVocab.from_t5()
decoder_vocab = DecoderVocab(tokenizer=NormalizedTokenizer())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(examples, model):
    tensor_examples = model.convert_numeric_examples(examples)
    return tensor_examples


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def _resolve_checkpoint_path(path_str: str) -> Optional[Path]:
    """checkpointmodel.pt"""
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
    if p.is_file():
        return p
    for fname in ["model.pt", "checkpoint.pt"]:
        direct_model = p / fname
        if direct_model.exists():
            return direct_model
    file_candidates = sorted(
        p.glob("checkpoint_*.pt"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    if file_candidates:
        return file_candidates[0]
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
            new_state = {}
            for key, value in state.items():
                new_state[f"module.{key}"] = value
            missing, unexpected = module.model.load_state_dict(new_state, strict=False)
        if unexpected:
            logger.info(f"checkpoint: {unexpected}")
        logger.info(f" {ckpt_path} ")
        return True
    except Exception as e:
        logger.error(f"checkpoint: {e}")
        return False


class RegressionModule(BaseModule):
    def __init__(self, cfg: DictConfig, accelerator: Optional[Accelerator] = None):
        self.accelerator = accelerator
        mlp_kwargs = {}
        if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type == "mlp":
            if hasattr(cfg.model, "input_dim"):
                mlp_kwargs["input_dim"] = cfg.model.input_dim
            if hasattr(cfg.model, "hidden_dims"):
                mlp_kwargs["hidden_dims"] = cfg.model.hidden_dims
            if hasattr(cfg.model, "output_dim"):
                mlp_kwargs["output_dim"] = cfg.model.output_dim

        model = torch_model_lib.PyTorchModel(
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            max_input_len=cfg.model.max_input_len,
            max_num_objs=cfg.model.max_num_objs,
            d_model=cfg.model.d_model,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            encoder_type=cfg.model.encoder_type,
            **mlp_kwargs,
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
        if hasattr(cfg, "reinforce") and cfg.reinforce.enabled:
            self.reinforce_loss_fn = ReinforceLoss(
                temperature=cfg.reinforce.get("temperature", 1.0),
                num_samples=cfg.reinforce.get("num_samples", 8),
                reward_scale=cfg.reinforce.get("reward_scale", 1.0),
                baseline_type=cfg.reinforce.get("baseline_type", "mean"),
            )
            self.reinforce_weight = cfg.reinforce.get("weight", 0.1)
            self.loss_balance = cfg.reinforce.get("loss_balance", False)
        else:
            self.reinforce_loss_fn = None
            self.reinforce_weight = 0.0

    def set_total_steps(self, total_steps: int):
        self.total_steps = total_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate
        )

        if not hasattr(self.cfg, "scheduler") or self.total_steps is None:
            return {"optimizer": optimizer}

        scheduler_config = self.cfg.scheduler
        scheduler_type = scheduler_config.get("type", "constant")

        if scheduler_type == "cosine":
            min_lr = self.cfg.learning_rate * scheduler_config.get("min_lr_ratio", 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - scheduler_config.get("warmup_steps", 0),
                eta_min=min_lr,
            )
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
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                )
        elif scheduler_type == "linear":
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
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        return {"optimizer": optimizer, "scheduler": scheduler}

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
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

                if self.reinforce_loss_fn is not None:
                    loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
                        batch,
                        self.NTL,
                        self.reinforce_loss_fn,
                        self.reinforce_weight,
                        self.loss_balance,
                    )
                else:
                    loss, metrics = self.model.compute_loss_and_metrics(batch, self.NTL)

                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                self.global_step += 1

        return {"train_loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        if not hasattr(self, "val_loader"):
            return {}

        self.model.eval()
        total_loss = 0

        progress_bar = accelerate_tqdm(
            self.val_loader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            if self.reinforce_loss_fn is not None:
                loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
                    batch, self.NTL, self.reinforce_loss_fn, self.reinforce_weight
                )
            else:
                loss, metrics = self.model.compute_loss_and_metrics(batch, self.NTL)
            total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        if self.best_save_metric == "val_loss":
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                if self.cfg.save_dir and self.accelerator.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        self, "best", {"val_loss": val_loss}
                    )

        return {"val_loss": val_loss}

    @torch.no_grad()
    def validate_epoch_with_r2(self, val_loader_for_r2=None) -> Dict[str, float]:
        """epochR2"""
        if not hasattr(self, "val_loader") and val_loader_for_r2 is None:
            return {}

        self.model.eval()
        total_loss = 0

        loader = val_loader_for_r2 if val_loader_for_r2 is not None else self.val_loader

        #
        progress_bar = accelerate_tqdm(
            loader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
        )

        all_predictions = []
        all_targets = []

        for batch in progress_bar:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            #
            if self.reinforce_loss_fn is not None:
                loss, metrics = self.model.compute_loss_and_metrics_with_reinforce(
                    batch, self.NTL, self.reinforce_loss_fn, self.reinforce_weight
                )
            else:
                loss, metrics = self.model.compute_loss_and_metrics(batch, self.NTL)
            total_loss += loss.item()

            # R2
            decoded_ids, output_floats = self.model.decode_with_mlp_encoder(
                batch, num_samples=32, temperature=1.0
            )
            output_floats = np.median(output_floats, axis=1)

            predictions = np.array(output_floats).flatten()
            targets = batch["y"].cpu().numpy().flatten()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

        val_loss = total_loss / len(loader)

        # R2
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        #
        y_max, y_min = load_train_stats(self.cfg.dataset.name, self.cfg.dataset.params.data_dir)
        if y_max is not None and y_min is not None:
            norm_pred, _, _ = normalize_data(predictions, y_max, y_min)
            norm_target, _, _ = normalize_data(targets, y_max, y_min)
        else:
            norm_pred = predictions
            norm_target = targets

        r2 = r2_score(norm_target, norm_pred)
        rank_corr, _ = spearmanr(norm_target, norm_pred)

        return {
            "val_loss": val_loss,
            "val_r2": r2,
            "val_rank_correlation": rank_corr,
        }

    def _fit_with_pruning(
        self, train_loader, num_epochs, val_loader, checkpoint_dir, save_every_n_epochs,
        trial: Optional[Trial] = None,
        optimize_r2: bool = False,
    ):
        """Optuna pruning"""
        if not hasattr(self, "train_loader"):
            self.prepare(train_loader, val_loader)

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

        for epoch in range(num_epochs):
            self.epoch_counter = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            metrics = {**train_metrics, **val_metrics}

            if self.logger:
                self.logger.log(metrics)

            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(f"Epoch [{epoch+1}/{num_epochs}]: {metrics_str}")

            # Optuna pruning - val_losspruning
            if trial is not None:
                trial.report(val_metrics.get("val_loss", float("inf")), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            #  - val_loss
            current_loss = val_metrics.get("val_loss", float("inf"))
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.early_stop_counter = 0
                if checkpoint_dir and self.accelerator.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        self, "best", {"val_loss": current_loss, **val_metrics}
                    )
            else:
                self.early_stop_counter += 1

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

        self.accelerator.end_training()

        # R2checkpoint
        # best_val_lossR2objective
        return self.best_val_loss


    def fit(
        self, train_loader, val_loader, num_epochs, checkpoint_dir, save_every_n_epochs,
        trial: Optional[Trial] = None,
        optimize_r2: bool = False,
    ):
        """"""
        weight_decay_flag = (
            getattr(self.cfg, "weight_decay_enable", False)
            if hasattr(self.cfg, "weight_decay_enable")
            else False
        )

        if not weight_decay_flag:
            return self._fit_with_pruning(
                train_loader=train_loader,
                num_epochs=num_epochs,
                val_loader=val_loader,
                checkpoint_dir=checkpoint_dir,
                save_every_n_epochs=save_every_n_epochs,
                trial=trial,
                optimize_r2=optimize_r2,
            )

        # weight_decay...
        # []
        if not hasattr(self, "train_loader"):
            self.prepare(train_loader, val_loader)

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
        initial_reinforce_weight = self.reinforce_weight

        for epoch in range(num_epochs):
            self.epoch_counter = epoch

            progress = min(1.0, epoch / max(1, num_epochs - 1))
            self.reinforce_weight = (
                initial_reinforce_weight + (1.0 - initial_reinforce_weight) * progress
            )

            if (epoch + 1) % 10 == 0:
                self.accelerator.print(
                    f"Epoch {epoch+1}: reinforce_weight = {self.reinforce_weight:.4f}"
                )

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            metrics = {**train_metrics, **val_metrics}

            if self.logger:
                self.logger.log({"reinforce_weight": self.reinforce_weight})
                self.logger.log(metrics)

            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(
                f"Epoch [{epoch+1}/{num_epochs}]: {metrics_str}, reinforce_weight: {self.reinforce_weight:.4f}"
            )

            # Optuna pruning - val_loss
            if trial is not None:
                trial.report(val_metrics.get("val_loss", float("inf")), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            #  - val_loss
            current_loss = val_metrics.get("val_loss", float("inf"))
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.early_stop_counter = 0
                if self.cfg.save_dir and self.accelerator.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        self, "best", {"val_loss": current_loss, **val_metrics}
                    )
            else:
                self.early_stop_counter += 1

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

        self.accelerator.end_training()

        return self.best_val_loss

    @torch.no_grad()
    def test_dataset(
        self, test_loader
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        self.model.eval()
        y_max, y_min = load_train_stats(self.cfg.dataset.name, self.cfg.dataset.params.data_dir)

        all_predictions = []
        all_targets = []

        for batch in test_loader:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            decoded_ids, output_floats = self.model.decode_with_mlp_encoder(
                batch, num_samples=128, temperature=1.0
            )

            output_floats = np.mean(output_floats, axis=1)

            predictions = np.array(output_floats).flatten()
            targets = batch["y"].cpu().numpy().flatten()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        norm_pred, _, _ = normalize_data(predictions, y_max, y_min)
        norm_target, _, _ = normalize_data(targets, y_max, y_min)

        mse = mean_squared_error(norm_target, norm_pred)
        rank_corr, _ = spearmanr(norm_target, norm_pred)
        r2 = r2_score(norm_target, norm_pred)

        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": np.mean(np.abs(norm_target - norm_pred)),
            "rank_correlation": rank_corr,
            "r2": r2,
        }

        return norm_pred, norm_target, metrics


def load_train_stats(dataset_name, regression_data_dir):
    train_y_path = os.path.join(regression_data_dir, dataset_name, "y_train.npy")
    try:
        if os.path.exists(train_y_path):
            train_y = np.load(train_y_path, allow_pickle=True)
            mean_val = np.mean(train_y)
            std_val = np.std(train_y)
            if std_val == 0:
                std_val = 1e-8
            transform_y = (train_y - mean_val) / std_val
            y_max = np.max(transform_y)
            y_min = np.min(transform_y)
            return y_max, y_min
        else:
            print(f": {dataset_name} y_train.npy{regression_data_dir}")
            return None, None
    except Exception as e:
        print(f"{dataset_name}: {str(e)}")
        return None, None


def normalize_data(data, y_max=None, y_min=None):
    data_array = np.array(data)
    normalized_data = data_array * (y_max - y_min + 1e-8) + y_min
    return normalized_data, y_max, y_min


def get_all_dataset_names(data_dir: str) -> List[str]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f": {data_dir}")

    dataset_names = []
    for item in data_path.iterdir():
        if item.is_dir() and (item / "info.json").exists():
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


def suggest_hyperparameters(trial: Trial, cfg: DictConfig) -> DictConfig:
    """
    Optuna trial
    """
    #
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # ============  ============

    #
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    cfg.base = trial.suggest_categorical("base", [2, 4, 6, 8, 10])
    cfg.digits = trial.suggest_categorical("digits", [4, 6, 8])

    #
    # cfg.batch_size = trial.suggest_categorical("batch_size", [128])

    #
    cfg.model.d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    cfg.model.nhead = trial.suggest_categorical("nhead", [4, 8])
    # cfg.model.num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    cfg.model.num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 5)
    cfg.model.dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
    # cfg.model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # MLP encoder
    if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type == "mlp":
        # hidden_dims
        num_hidden_layers = cfg.model.num_decoder_layers
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
        cfg.model.hidden_dims = [hidden_dim] * num_hidden_layers
        cfg.model.output_dim = cfg.model.d_model

    return cfg


def save_best_params_to_checkpoint(
    best_params: Dict,
    checkpoint_dir: Path,
    study_stats: Dict = None,
):
    """
    checkpointRL
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    #
    best_params_file = checkpoint_dir / "best_hyperparams.json"
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f": {best_params_file}")

    #
    if study_stats is not None:
        stats_file = checkpoint_dir / "optuna_study_stats.json"
        with open(stats_file, "w") as f:
            json.dump(study_stats, f, indent=2)
        logger.info(f"Optuna: {stats_file}")

    return best_params_file


def load_best_params_from_checkpoint(checkpoint_dir: Path) -> Optional[Dict]:
    """
    checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_params_file = checkpoint_dir / "best_hyperparams.json"

    if best_params_file.exists():
        with open(best_params_file, "r") as f:
            best_params = json.load(f)
        logger.info(f" {best_params_file} ")
        return best_params
    else:
        logger.warning(f": {best_params_file}")
        return None


def apply_best_params_to_config(cfg: DictConfig, best_params: Dict) -> DictConfig:
    """

    """
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    for key, value in best_params.items():
        if key == "learning_rate":
            cfg.learning_rate = value
        elif key == "batch_size":
            cfg.batch_size = value
        elif key == "d_model":
            cfg.model.d_model = value
        elif key == "nhead":
            cfg.model.nhead = value
        elif key == "num_encoder_layers":
            cfg.model.num_encoder_layers = value
        elif key == "num_decoder_layers":
            cfg.model.num_decoder_layers = value
        elif key == "dim_feedforward":
            cfg.model.dim_feedforward = value
        elif key == "dropout":
            cfg.model.dropout = value
        elif key == "num_hidden_layers":
            hidden_dim = best_params.get("hidden_dim", 256)
            cfg.model.hidden_dims = [hidden_dim] * value
        elif key == "hidden_dim":
            num_layers = best_params.get("num_hidden_layers", 2)
            cfg.model.hidden_dims = [value] * num_layers
        elif key == "mlp_output_dim":
            cfg.model.output_dim = value
        elif key.startswith("reinforce_"):
            param_name = key.replace("reinforce_", "")
            if hasattr(cfg, "reinforce"):
                cfg.reinforce[param_name] = value
        elif key == "scheduler_type":
            if hasattr(cfg, "scheduler"):
                cfg.scheduler.type = value
        elif key in ["warmup_steps", "min_lr_ratio"]:
            if hasattr(cfg, "scheduler"):
                cfg.scheduler[key] = value

    return cfg


def create_objective(base_cfg: DictConfig, dataset_name: str, results_dir: Path, accelerator: Optional[Accelerator] = None):
    """
    Optuna - R2
    """
    def objective(trial: Trial) -> float:
        """Optuna - R2"""
        # accelerator.free_memory()
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        try:
            #
            cfg = suggest_hyperparameters(trial, base_cfg)

            #
            cfg.dataset.name = dataset_name
            cfg.experiment_name = f"{base_cfg.experiment_name}_{dataset_name}_trial_{trial.number}"
            cfg.save_dir = str(results_dir / dataset_name / f"trial_{trial.number}" / "checkpoints")

            # wandb
            cfg.use_wandb = False

            os.makedirs(cfg.save_dir, exist_ok=True)

            #
            seed_everything(cfg.seed)

            #
            train_dataset = Binary_fit_Dataset(
                data_dir=cfg.dataset.params.data_dir,
                split="train",
                dataset_name=dataset_name,
            )
            val_dataset = Binary_fit_Dataset(
                data_dir=cfg.dataset.params.data_dir,
                split="val",
                dataset_name=dataset_name,
            )
            test_dataset = Binary_fit_Dataset(
                data_dir=cfg.dataset.params.data_dir,
                split="test",
                dataset_name=dataset_name,
            )

            #
            task_dim = train_dataset.dimension
            if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type == "mlp":
                cfg.model.input_dim = task_dim

            #
            global decoder_vocab
            decoder_vocab = DecoderVocab(tokenizer=NormalizedTokenizer(num_digits=cfg.digits, base=cfg.base))
            module = RegressionModule(cfg, accelerator=accelerator)
            custom_collate = lambda examples: collate_fn(examples, module.model)

            #
            if hasattr(cfg, "init_checkpoint"):
                resolved = _resolve_checkpoint_path(cfg.init_checkpoint)
                if resolved is not None:
                    _load_checkpoint_into_module(module, resolved)

            #
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=custom_collate,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=custom_collate,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=16,
                shuffle=False,
                collate_fn=custom_collate,
            )

            #
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * cfg.num_epochs
            module.set_total_steps(total_steps)

            # val_losspruning
            best_val_loss = module.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=cfg.num_epochs,
                checkpoint_dir=cfg.save_dir,
                save_every_n_epochs=cfg.save_every_n_epochs,
                trial=trial,
                optimize_r2=True,  # R2
            )

            # ===== checkpointR2 =====
            best_ckpt_path = cfg.save_dir + "/checkpoint_best/model.pt"
            resolved = _resolve_checkpoint_path(best_ckpt_path)
            if resolved is not None:
                success = _load_checkpoint_into_module(module, resolved)
                if not success:
                    logger.warning(f"Trial {trial.number}: checkpoint")

            # R2
            predictions, targets, test_metrics = module.test_dataset(val_loader)

            test_r2 = test_metrics["r2"]

            logger.info(
                f"Trial {trial.number} completed - val_loss: {best_val_loss:.6f}, "
                f"test_r2: {test_r2:.6f}, test_mse: {test_metrics['mse']:.6f}"
            )

            # trial
            trial_results = {
                "trial_number": trial.number,
                "params": trial.params,
                "best_val_loss": best_val_loss,
                "test_r2": test_r2,
                "test_metrics": test_metrics,
            }
            trial_results_file = Path(cfg.save_dir).parent / "trial_results.json"
            with open(trial_results_file, "w") as f:
                json.dump(trial_results, f, indent=2, default=str)

            del module
            if 'optimizer' in locals():
                del optimizer
            if 'scheduler' in locals():
                del scheduler

            gc.collect()
            torch.cuda.empty_cache()

            return test_r2  # R2Optuna

        except optuna.TrialPruned:
            # Pruned
            if 'module' in locals(): del module
            gc.collect()
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            #
            if 'module' in locals(): del module
            gc.collect()
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned() #  e

    return objective


def run_optuna_search(
    cfg: DictConfig,
    dataset_name: str,
    results_dir: Path,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """
    Optuna - R2
    SQLite
    """
    accelerator = Accelerator(
            gradient_accumulation_steps=1,
        )
    if study_name is None:
        study_name = f"regression_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # SQLite
    if storage is None:
        db_path = results_dir / dataset_name
        db_path.mkdir(parents=True, exist_ok=True)
        # 1:  .resolve()  Hydra
        storage = f"sqlite:///{db_path.resolve()}/optuna_{dataset_name}.db"

    # pruner - R2
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    )

    # sampler
    sampler = optuna.samplers.TPESampler(
        seed=cfg.seed,
        n_startup_trials=10,
    )

    # 2:
    # SQLite
    study = None
    max_retries = 10

    for i in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",  # R2
                pruner=pruner,
                sampler=sampler,
                load_if_exists=load_if_exists,
            )
            break  #
        except (OperationalError, Exception) as e:
            #  "table studies already exists"  "database is locked"
            error_msg = str(e)
            if "already exists" in error_msg or "database is locked" in error_msg:
                if i < max_retries - 1:
                    wait_time = random.uniform(1, 5)  #  1-5
                    logger.warning(f"Optuna{wait_time:.2f} ({i+1}/{max_retries})... : {error_msg}")
                    time.sleep(wait_time)
                    continue
            raise e  #

    if study is None:
        raise RuntimeError(" Optuna Study")

    #  ( exp5_mlp_encoder_ce )
    initial_params = {
        "learning_rate": cfg.learning_rate,
        "base": cfg.base,
        "digits": cfg.digits,
        "d_model": cfg.model.d_model,
        "nhead": cfg.model.nhead,
        "num_decoder_layers": cfg.model.num_decoder_layers,
        "dim_feedforward": cfg.model.dim_feedforward,
    }
    if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type == "mlp":
        if hasattr(cfg.model, "hidden_dims"):
            hidden_dims = cfg.model.hidden_dims
            if isinstance(hidden_dims, (list, List)) and len(hidden_dims) > 0:
                initial_params["hidden_dim"] = hidden_dims[0]
            elif isinstance(hidden_dims, int):
                initial_params["hidden_dim"] = hidden_dims

    study.enqueue_trial(initial_params)
    logger.info(f" Trial: {initial_params}")

    #
    objective = create_objective(cfg, dataset_name, results_dir, accelerator=accelerator)

    #
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    return study


def train_with_best_params(
    cfg: DictConfig,
    dataset_name: str,
    results_dir: Path,
    best_params: Dict,
) -> Dict:
    """"""
    logger.info(f": {best_params}")

    #
    cfg = apply_best_params_to_config(cfg, best_params)

    cfg.dataset.name = dataset_name
    cfg.experiment_name = f"{cfg.experiment_name}_{dataset_name}_best"

    #  - RL
    best_checkpoint_dir = results_dir / dataset_name / "best_model" / "checkpoints"
    cfg.save_dir = str(best_checkpoint_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # val_losstest
    result = train_and_test_single_task(
        cfg,
        dataset_name,
        results_dir / dataset_name / "best_model",
    )

    # R2checkpoint
    save_best_params_to_checkpoint(
        best_params=best_params,
        checkpoint_dir=best_checkpoint_dir,
        study_stats={
            "best_test_r2": result["metrics"]["r2"],
            "best_test_mse": result["metrics"]["mse"],
            "best_test_rank_correlation": result["metrics"]["rank_correlation"],
            "best_params": best_params,
        }
    )

    return result


def train_and_test_single_task(
    cfg: DictConfig,
    dataset_name: str,
    results_dir: Path,
    optimize_r2: bool = False,
    accelerator: Optional[Accelerator] = None,
) -> Dict:
    """"""
    logger.info(f": {dataset_name}")

    task_cfg = cfg.copy() if hasattr(cfg, 'copy') else OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    task_cfg.dataset.params.data_dir = cfg.dataset.params.data_dir
    task_cfg.experiment_name = f"{cfg.experiment_name}_{dataset_name}"
    task_cfg.save_dir = str(results_dir / "checkpoints")

    os.makedirs(task_cfg.save_dir, exist_ok=True)

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
    logger.info(f" {dataset_name} : {task_dim}")

    if hasattr(task_cfg.model, "encoder_type") and task_cfg.model.encoder_type == "mlp":
        task_cfg.model.input_dim = task_dim
        logger.info(f"MLP encoderinput_dim: {task_dim}")

    module = RegressionModule(task_cfg, accelerator=accelerator)
    custom_collate = lambda examples: collate_fn(examples, module.model)

    init_ckpt = None
    if hasattr(task_cfg, "init_checkpoint"):
        init_ckpt = task_cfg.init_checkpoint
    if init_ckpt:
        resolved = _resolve_checkpoint_path(init_ckpt)
        if resolved is not None:
            _load_checkpoint_into_module(module, resolved)

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
        batch_size=16,
        shuffle=False,
        collate_fn=custom_collate,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * task_cfg.num_epochs
    module.set_total_steps(total_steps)

    logger.info(
        f" {dataset_name}: {steps_per_epoch} steps/epoch x {task_cfg.num_epochs} epochs = {total_steps} total steps"
    )

    module.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=task_cfg.num_epochs,
        checkpoint_dir=task_cfg.save_dir,
        save_every_n_epochs=task_cfg.save_every_n_epochs,
        optimize_r2=optimize_r2,
    )

    #
    ckpt_path = task_cfg.save_dir + "/checkpoint_best/model.pt"
    resolved = _resolve_checkpoint_path(ckpt_path)
    if resolved is not None:
        success = _load_checkpoint_into_module(module, resolved)
        if not success:
            logger.warning("checkpoint")

    predictions, targets, metrics = module.test_dataset(test_loader)

    task_results = {
        "dataset_name": dataset_name,
        "metrics": metrics,
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
    }

    results_file = results_dir / f"results_{cfg.seed}.json"
    with open(results_file, "w") as f:
        json.dump(task_results, f, indent=2)

    logger.info(
        f" {dataset_name}  - R2: {metrics['r2']:.6f}, MSE: {metrics['mse']:.6f}, Rank Corr: {metrics['rank_correlation']:.6f}"
    )

    return task_results


def save_optuna_results(study: optuna.Study, results_dir: Path, dataset_name: str):
    """Optuna"""
    output_dir = results_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    #
    best_params_file = output_dir / "best_params.json"
    with open(best_params_file, "w") as f:
        json.dump(study.best_params, f, indent=2)

    #
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "all_trials.csv", index=False)

    #
    study_stats = {
        "best_value": study.best_value,  # R2
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "optimization_direction": "maximize",
        "optimization_metric": "R2",
    }
    with open(output_dir / "study_stats.json", "w") as f:
        json.dump(study_stats, f, indent=2)

    logger.info(f"Optuna: {output_dir}")
    logger.info(f"R2: {study.best_value:.6f}")
    logger.info(f": {study.best_params}")


@hydra.main(config_path="../conf", config_name="config_mlp_example", version_base=None)
def main(cfg: DictConfig):
    """OptunaR2"""

    #
    dataset_name = cfg.dataset.name

    #
    seed_everything(cfg.seed)

    #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"outputs/optuna_ce/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("skip_mode", False):
        results_json_path = Path(f"./outputs/optuna_ce/{dataset_name}/{dataset_name}/trial_0/checkpoints/checkpoint_best/metrics.json")
        if results_json_path.exists():
            logger.info(f"{dataset_name} ")
            return

    # Optuna
    use_optuna = cfg.get("use_optuna", True)

    if use_optuna:
        logger.info(f" {dataset_name} OptunaR2")

        # Optuna
        optuna_cfg = cfg.get("optuna", {})
        n_trials = optuna_cfg.get("n_trials", 25)
        timeout = optuna_cfg.get("timeout", None)

        # Optuna
        study = run_optuna_search(
            cfg=cfg,
            dataset_name=dataset_name,
            results_dir=results_dir,
            n_trials=n_trials,
            timeout=timeout,
        )

        #
        save_optuna_results(study, results_dir, dataset_name)

        #
        logger.info(f"R2: {study.best_value:.6f}")
        logger.info(f": {study.best_params}")

        #
        if cfg.get("train_with_best", True):
            logger.info("...")
            if cfg.use_wandb:
                swanlab.init(project=cfg.project_name, name=f"{cfg.experiment_name}_best")

            # final_result = train_with_best_params(
            #     cfg=cfg,
            #     dataset_name=dataset_name,
            #     results_dir=results_dir,
            #     best_params=study.best_params,
            # )

            # logger.info(f" - R2: {final_result['metrics']['r2']:.6f}")
            # logger.info(f" - MSE: {final_result['metrics']['mse']:.6f}")
            # logger.info(f" - Rank Correlation: {final_result['metrics']['rank_correlation']:.6f}")

            # checkpointRL
            best_ckpt_dir = results_dir / dataset_name / "best_model" / "checkpoints"
            logger.info(f"=" * 60)
            logger.info(f"checkpoint: {best_ckpt_dir}")
            logger.info(f": {best_ckpt_dir / 'best_hyperparams.json'}")
            logger.info(f": {best_ckpt_dir / 'checkpoint_best' / 'model.pt'}")
            logger.info(f"=" * 60)

    else:
        #
        if cfg.use_wandb:
            swanlab.init(project=cfg.project_name, name=cfg.experiment_name)

        logger.info(f" {dataset_name} ")

        if cfg.get("skip_mode", False):
            results_json_path = Path(f"./outputs/optuna_ce/{dataset_name}/{dataset_name}/trial_0/checkpoints/checkpoint_best/metrics.json")
            if results_json_path.exists():
                logger.info(f"{dataset_name} ")
                return

        result = train_and_test_single_task(
            cfg,
            dataset_name,
            results_dir,
            optimize_r2=cfg.get("optimize_r2", False),
        )

        logger.info(f" {dataset_name} : {results_dir}")
        logger.info(f"R2: {result['metrics']['r2']:.6f}")
        logger.info(f"MSE: {result['metrics']['mse']:.6f}")
        logger.info(f"Rank Correlation: {result['metrics']['rank_correlation']:.6f}")


# ============ RL ============

def load_best_config_for_rl(
    base_cfg: DictConfig,
    dataset_name: str,
    optuna_results_dir: str = "outputs/optuna",
) -> Tuple[DictConfig, Path]:
    """
    RL

    Args:
        base_cfg:
        dataset_name:
        optuna_results_dir: Optuna

    Returns:
        Tuple[DictConfig, Path]: checkpoint
    """
    # checkpoint
    best_ckpt_dir = Path(optuna_results_dir) / dataset_name / dataset_name / "best_model" / "checkpoints"

    #
    best_params = load_best_params_from_checkpoint(best_ckpt_dir)

    if best_params is None:
        logger.warning(f" {dataset_name} ")
        return base_cfg, None

    #
    updated_cfg = apply_best_params_to_config(base_cfg, best_params)

    #
    model_path = best_ckpt_dir / "checkpoint_best" / "model.pt"

    return updated_cfg, model_path


if __name__ == "__main__":
    main()