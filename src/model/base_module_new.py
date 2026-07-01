import abc
from functools import partial
from typing import Dict, Optional, List, Tuple

import swanlab
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm
from torch.utils.data import DataLoader

from src.utils.checkpoint import CheckpointManager
from src.utils.number_token_loss import NumberTokenLoss
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

import os
import numpy as np
from pathlib import Path

class BaseModule:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        gradient_accumulation_steps: int = 1,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        log_dir: Optional[str] = None,
    ) -> None:
        # self.accelerator = Accelerator(
        #     gradient_accumulation_steps=gradient_accumulation_steps,
        # )

        self.model = model
        self.criterion = criterion
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.val_global_step = 0

        self.model_params = self._count_parameters()
        self.accelerator.print(f"Model parameters: {self.model_params:,}")

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        self.logger = None
        if self.use_wandb and self.project_name:
            self.logger = swanlab

    def _count_parameters(self) -> int:
        """

        Returns:
            int:
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params

    def get_model_parameters(self) -> Dict[str, int]:
        """

        Returns:
            Dict[str, int]:
        """
        return {
            "total_parameters": self.model_params,
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "non_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if not p.requires_grad
            ),
        }

    @abc.abstractmethod
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            torch.optim.Optimizer or dict: Optimizer or dict containing optimizer and scheduler
        """
        pass

    def prepare(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        if self.optimizer is None:
            optimizer_dict = self.configure_optimizers()
            if isinstance(optimizer_dict, dict):
                self.optimizer = optimizer_dict["optimizer"]
                self.scheduler = optimizer_dict.get("scheduler")

        if val_loader is not None:
            (
                self.model,
                self.optimizer,
                train_loader,
                val_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader, self.scheduler
            )
            self.val_loader = val_loader
        else:
            (
                self.model,
                self.optimizer,
                train_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, self.scheduler
            )

        self.train_loader = train_loader

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        progress_bar = accelerate_tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                    targets = batch.get("labels") or batch.get("targets")
                    if targets is None:
                        loss = outputs.loss
                    else:
                        if self.criterion:
                            loss = self.criterion(outputs, targets)
                        else:
                            loss, _ = self.model.compute_loss_and_metrics(
                                batch, self.NTL
                            )
                else:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if self.logger:
                    self.logger.log({"train_step_loss": loss.item()})
                self.global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                targets = batch.get("labels") or batch.get("targets")
                if targets is None:
                    loss = outputs.loss
                else:
                    if self.criterion:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss, _ = self.model.compute_loss_and_metrics(batch)
            else:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = total_loss / len(self.val_loader)

        return {"val_loss": val_loss}

    def save_checkpoint(self, save_path: str, epoch: int = None, metrics: Dict = None):
        self.accelerator.save_state(f"{save_path}/ckpt_{epoch}")

        import json

        param_info = self.get_model_parameters()
        param_file = f"{save_path}/model_parameters.json"
        with open(param_file, "w") as f:
            json.dump(param_info, f, indent=2)

    def load_checkpoint(self, load_path: str) -> Dict:
        self.accelerator.load_state(load_path)

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[list] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
    ):
        if not hasattr(self, "train_loader"):
            self.prepare(train_loader, val_loader)

        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        if self.logger:
            param_info = self.get_model_parameters()
            self.logger.log(param_info)
            self.accelerator.print(f"Model parameters logged: {param_info}")

        self.best_val_loss = float("inf")
        self.best_val_mse = float("inf")
        self.early_stop_counter = 0
        self.early_stop_patience = 20
        self.epoch_counter = 0

        for epoch in range(num_epochs):
            self.epoch_counter = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            metrics = {**train_metrics, **val_metrics}

            if self.logger:
                self.logger.log(metrics)

            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(f"Epoch [{epoch+1}/{num_epochs}]: {metrics_str}")

            if self.accelerator.is_main_process and checkpoint_dir:
                if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                    self.checkpoint_manager.save_checkpoint(self, epoch + 1, metrics)

            if callbacks:
                for callback in callbacks:
                    callback(self, metrics, epoch)

            if self.early_stop_counter >= self.early_stop_patience:
                self.accelerator.print("Early stopping triggered")
                self.save_checkpoint(checkpoint_dir, epoch, metrics)
                break

        if self.accelerator.is_main_process and checkpoint_dir:
            self.checkpoint_manager.save_checkpoint(self, num_epochs, metrics)

        self.accelerator.end_training()

    @torch.no_grad()
    def test_dataset(
        self, test_loader
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """"""
        self.model.eval()
        train_mean, train_std, max_val, min_val = load_train_stats(self.cfg.dataset.name,self.cfg.dataset.params.data_dir)
        all_predictions_mean = []
        all_predictions_median = []
        all_predictions_clip_mean = []
        all_predictions_clip_median = []
        all_targets = []
        # sensible_predictions = 0

        for batch in test_loader:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            #
            decoded_ids, output_floats = self.model.decode_with_mlp_encoder(
                batch, num_samples=128, temperature=1.0  #
            )
            #
            # lower = train_mean - 3 * train_std
            # upper = train_mean + 3 * train_std
            # output_floats+-3*
            # output_floats: (batch_size, 128)
            # within_range = ((output_floats >= lower) & (output_floats <= upper)).sum(axis=1)
            # sensible_predictions += within_range.sum()

            # output_floats
            norm_output_floats, _, _ = normalize_data(output_floats, train_mean, train_std)

            norm_output_floats_clip = np.clip(norm_output_floats, 1.1*min_val, 1.1*max_val)

            output_floats_mean = np.mean(norm_output_floats, axis=1)
            output_floats_median = np.median(norm_output_floats, axis=1)
            output_floats_clip_mean = np.mean(norm_output_floats_clip, axis=1)
            output_floats_clip_median = np.median(norm_output_floats_clip, axis=1)
            #
            predictions_mean = np.array(output_floats_mean).flatten()
            predictions_median = np.array(output_floats_median).flatten()
            predictions_clip_mean = np.array(output_floats_clip_mean).flatten()
            predictions_clip_median = np.array(output_floats_clip_median).flatten()
            targets = batch["y"].cpu().numpy().flatten()

            all_predictions_mean.extend(predictions_mean)
            all_predictions_median.extend(predictions_median)
            all_predictions_clip_mean.extend(predictions_clip_mean)
            all_predictions_clip_median.extend(predictions_clip_median)
            all_targets.extend(targets)

        predictions_mean = np.array(all_predictions_mean)
        predictions_median = np.array(all_predictions_median)
        predictions_clip_mean = np.array(all_predictions_clip_mean)
        predictions_clip_median = np.array(all_predictions_clip_median)
        targets = np.array(all_targets)
        norm_target, _, _ = normalize_data(targets, train_mean, train_std)  # targets

        mse_mean = mean_squared_error(norm_target, predictions_mean)
        rank_corr_mean, _ = spearmanr(norm_target, predictions_mean)
        r2_mean = r2_score(norm_target, predictions_mean)
        mse_median = mean_squared_error(norm_target, predictions_median)
        rank_corr_median, _ = spearmanr(norm_target, predictions_median)
        r2_median = r2_score(norm_target, predictions_median)
        mse_clip_mean = mean_squared_error(norm_target, predictions_clip_mean)
        rank_corr_clip_mean, _ = spearmanr(norm_target, predictions_clip_mean)
        r2_clip_mean = r2_score(norm_target, predictions_clip_mean)
        mse_clip_median = mean_squared_error(norm_target, predictions_clip_median)
        rank_corr_clip_median, _ = spearmanr(norm_target, predictions_clip_median)
        r2_clip_median = r2_score(norm_target, predictions_clip_median)
        # total_samples = len(all_targets) * 128  # target128samples
        # sensible_ratio = sensible_predictions / total_samples if total_samples > 0 else 0
        # print(f"sensible ratio (mean+-3sigma): {sensible_ratio:.4f}")
        metrics_mean = {
            "mse": mse_mean,
            "rmse": np.sqrt(mse_mean),
            "mae": np.mean(np.abs(norm_target - predictions_mean)),
            "rank_correlation": rank_corr_mean,
            "r2": r2_mean,
            # "sensible_ratio": sensible_ratio,
        }
        metrics_median = {
            "mse": mse_median,
            "rmse": np.sqrt(mse_median),
            "mae": np.mean(np.abs(norm_target - predictions_median)),
            "rank_correlation": rank_corr_median,
            "r2": r2_median,
        }
        metrics_clip_mean = {
            "mse": mse_clip_mean,
            "rmse": np.sqrt(mse_clip_mean),
            "mae": np.mean(np.abs(norm_target - predictions_clip_mean)),
            "rank_correlation": rank_corr_clip_mean,
            "r2": r2_clip_mean,
        }
        metrics_clip_median = {
            "mse": mse_clip_median,
            "rmse": np.sqrt(mse_clip_median),
            "mae": np.mean(np.abs(norm_target - predictions_clip_median)),
            "rank_correlation": rank_corr_clip_median,
            "r2": r2_clip_median,
        }
        return predictions_mean, predictions_median, predictions_clip_mean, predictions_clip_median, norm_target, metrics_mean, metrics_median, metrics_clip_mean, metrics_clip_median

    @torch.no_grad()
    def test_dataset_normalized(
        self, test_loader
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """"""
        self.model.eval()
        y_max, y_min = load_train_stats_normalized(self.cfg.dataset.name,self.cfg.dataset.params.data_dir)
        all_predictions_mean = []
        all_predictions_median = []
        all_predictions_clip_mean = []
        all_predictions_clip_median = []
        all_targets = []
        # sensible_predictions = 0

        for batch in test_loader:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            #
            decoded_ids, output_floats = self.model.decode_with_mlp_encoder(
                batch, num_samples=128, temperature=1.0  #
            )
            #
            # lower = train_mean - 3 * train_std
            # upper = train_mean + 3 * train_std
            # output_floats+-3*
            # output_floats: (batch_size, 128)
            # within_range = ((output_floats >= lower) & (output_floats <= upper)).sum(axis=1)
            # sensible_predictions += within_range.sum()

            # output_floats
            norm_output_floats, _, _ = normalize_data_normalized(output_floats, y_max, y_min)

            norm_output_floats_clip = np.clip(norm_output_floats, 1.1*y_min, 1.1*y_max)

            output_floats_mean = np.mean(norm_output_floats, axis=1)
            output_floats_median = np.median(norm_output_floats, axis=1)
            output_floats_clip_mean = np.mean(norm_output_floats_clip, axis=1)
            output_floats_clip_median = np.median(norm_output_floats_clip, axis=1)
            #
            predictions_mean = np.array(output_floats_mean).flatten()
            predictions_median = np.array(output_floats_median).flatten()
            predictions_clip_mean = np.array(output_floats_clip_mean).flatten()
            predictions_clip_median = np.array(output_floats_clip_median).flatten()
            targets = batch["y"].cpu().numpy().flatten()

            all_predictions_mean.extend(predictions_mean)
            all_predictions_median.extend(predictions_median)
            all_predictions_clip_mean.extend(predictions_clip_mean)
            all_predictions_clip_median.extend(predictions_clip_median)
            all_targets.extend(targets)

        predictions_mean = np.array(all_predictions_mean)
        predictions_median = np.array(all_predictions_median)
        predictions_clip_mean = np.array(all_predictions_clip_mean)
        predictions_clip_median = np.array(all_predictions_clip_median)
        targets = np.array(all_targets)
        norm_target, _, _ = normalize_data_normalized(targets, y_max, y_min)  # targets

        mse_mean = mean_squared_error(norm_target, predictions_mean)
        rank_corr_mean, _ = spearmanr(norm_target, predictions_mean)
        r2_mean = r2_score(norm_target, predictions_mean)
        mse_median = mean_squared_error(norm_target, predictions_median)
        rank_corr_median, _ = spearmanr(norm_target, predictions_median)
        r2_median = r2_score(norm_target, predictions_median)
        mse_clip_mean = mean_squared_error(norm_target, predictions_clip_mean)
        rank_corr_clip_mean, _ = spearmanr(norm_target, predictions_clip_mean)
        r2_clip_mean = r2_score(norm_target, predictions_clip_mean)
        mse_clip_median = mean_squared_error(norm_target, predictions_clip_median)
        rank_corr_clip_median, _ = spearmanr(norm_target, predictions_clip_median)
        r2_clip_median = r2_score(norm_target, predictions_clip_median)
        # total_samples = len(all_targets) * 128  # target128samples
        # sensible_ratio = sensible_predictions / total_samples if total_samples > 0 else 0
        # print(f"sensible ratio (mean+-3sigma): {sensible_ratio:.4f}")
        metrics_mean = {
            "mse": mse_mean,
            "rmse": np.sqrt(mse_mean),
            "mae": np.mean(np.abs(norm_target - predictions_mean)),
            "rank_correlation": rank_corr_mean,
            "r2": r2_mean,
            # "sensible_ratio": sensible_ratio,
        }
        metrics_median = {
            "mse": mse_median,
            "rmse": np.sqrt(mse_median),
            "mae": np.mean(np.abs(norm_target - predictions_median)),
            "rank_correlation": rank_corr_median,
            "r2": r2_median,
        }
        metrics_clip_mean = {
            "mse": mse_clip_mean,
            "rmse": np.sqrt(mse_clip_mean),
            "mae": np.mean(np.abs(norm_target - predictions_clip_mean)),
            "rank_correlation": rank_corr_clip_mean,
            "r2": r2_clip_mean,
        }
        metrics_clip_median = {
            "mse": mse_clip_median,
            "rmse": np.sqrt(mse_clip_median),
            "mae": np.mean(np.abs(norm_target - predictions_clip_median)),
            "rank_correlation": rank_corr_clip_median,
            "r2": r2_clip_median,
        }
        return predictions_mean, predictions_median, predictions_clip_mean, predictions_clip_median, norm_target, metrics_mean, metrics_median, metrics_clip_mean, metrics_clip_median

def load_train_stats(dataset_name, regression_data_dir):
    """
    regression_datay_train.npy
    """
    train_y_path = os.path.join(regression_data_dir, dataset_name, "y_train.npy")
    try:
        if os.path.exists(train_y_path):
            train_y = np.load(train_y_path, allow_pickle=True)
            mean_val = np.mean(train_y)
            std_val = np.std(train_y)
            norm_y = (train_y - mean_val) / std_val
            max_val = np.max(norm_y)
            min_val = np.min(norm_y)
            if std_val == 0:
                std_val = 1e-8
            # print("meanstd")
            return mean_val, std_val, max_val, min_val
        else:
            print(f": {dataset_name} y_train.npy{regression_data_dir}")
            return None, None, None, None
    except Exception as e:
        print(f"{dataset_name}: {str(e)}")
        return None, None, None, None

def load_train_stats_normalized(dataset_name, regression_data_dir):
    """
    regression_datay_train.npy
    """
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
            # print("meanstd")
            return y_max, y_min
        else:
            print(f": {dataset_name} y_train.npy{regression_data_dir}")
            return None, None
    except Exception as e:
        print(f"{dataset_name}: {str(e)}")
        return None, None

def normalize_data_normalized(data, y_max=None, y_min=None):
    """


    """
    data_array = np.array(data)

    normalized_data = data_array * (y_max - y_min + 1e-8) + y_min
    # print("")
    return normalized_data, y_max, y_min


def normalize_data(data, mean_val=None, std_val=None):
    """


    """
    data_array = np.array(data)

    if mean_val is not None and std_val is not None:
        #
        normalized_data = (data_array - mean_val) / std_val
        # print("")
        return normalized_data, mean_val, std_val
    else:
        #
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val == 0:
            std_val = 1e-8
        normalized_data = (data_array - mean_val) / std_val
        return normalized_data.tolist(), mean_val, std_val
