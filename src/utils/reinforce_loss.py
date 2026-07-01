from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ReinforceLoss:
    """REINFORCE"""

    def __init__(
        self,
        temperature: float = 1.0,
        num_samples: int = 8,
        reward_scale: float = 1.0,
        baseline_type: str = "mean",
        kernel_sigma: float = 1.0,
        kernel_reduction: str = "mean",
    ):
        """
        REINFORCE

        Args:
            temperature:
            num_samples:
            reward_scale: reward
            baseline_type: baseline'mean''min'
            kernel_sigma: sigma
            kernel_reduction: ()'mean''sum'
        """
        self.temperature = temperature
        self.num_samples = num_samples
        self.reward_scale = reward_scale
        self.baseline_type = baseline_type
        self.kernel_sigma = kernel_sigma
        self.kernel_reduction = kernel_reduction

    def compute_reward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        rewardexp(-||y_hat - y||^2 / (2*sigma^2))

         (B,S)  (B,S,D)kernel_reduction

        Args:
            predictions:  (B, S)  (B, S, D)
            targets:  (B,)
        Returns:
            rewards:  (B, S)
        """
        sigma2 = float(self.kernel_sigma) ** 2 + 1e-12
        if predictions.dim() == 3:
            # (B, S, D)
            targets_expanded = (
                targets.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, predictions.size(1), predictions.size(2))
            )
            diff2 = (predictions - targets_expanded) ** 2  # (B, S, D)
            if self.kernel_reduction == "sum":
                dist2 = diff2.sum(dim=2)
            else:
                dist2 = diff2.mean(dim=2)
        else:
            # (B, S)
            targets_expanded = targets.unsqueeze(1).expand(-1, predictions.size(1))
            dist2 = (predictions - targets_expanded) ** 2  # (B, S)
        rewards = torch.exp(-0.5 * dist2 / sigma2) * self.reward_scale  # (B, S)
        return rewards

    def compute_baseline(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        baseline

        Args:
            rewards: reward shape (batch_size, num_samples)

        Returns:
            baseline: baseline shape (batch_size,)
        """
        if self.baseline_type == "mean":
            baseline = rewards.mean(dim=1)
        elif self.baseline_type == "min":
            baseline = rewards.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

        return baseline

    def compute_reinforce_loss(
        self, log_probs: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        REINFORCE

        Args:
            log_probs:  shape (batch_size, num_samples, seq_len)
            rewards: reward shape (batch_size, num_samples)

        Returns:
            loss: REINFORCE
        """
        batch_size, num_samples, seq_len = log_probs.shape

        #
        # log_probs: (batch_size, num_samples, seq_len) -> (batch_size, num_samples)
        sequence_log_probs = log_probs.sum(dim=2)

        # baseline
        baseline = self.compute_baseline(rewards)  # (batch_size,)
        baseline_expanded = baseline.unsqueeze(1).expand(
            -1, num_samples
        )  # (batch_size, num_samples)

        # advantage
        advantage = rewards - baseline_expanded  # (batch_size, num_samples)
        # REINFORCE
        # rewardlog_prob * advantage
        reinforce_loss = -(sequence_log_probs * advantage).mean()

        return reinforce_loss

    def __call__(
        self, model, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        REINFORCE

        Args:
            model:
            batch: batch

        Returns:
            loss:
            metrics:
        """
        batch_size = batch["encoder_input"].shape[0]
        device = batch["encoder_input"].device

        # log prob
        # log prob no_grad
        decoded_ids, output_floats, step_log_probs = model.sample_with_logprobs(
            batch, self.num_samples, self.temperature
        )

        # numpytensor
        predictions = torch.tensor(
            output_floats, device=device, dtype=torch.float32
        )  # (batch_size, num_samples)
        targets = batch["y"]  # (batch_size,)
        y_medians = batch["y_median"]
        q1s = batch["q1"]
        q3s = batch["q3"]

        predictions = (predictions - y_medians) / (q3s - q1s)
        targets = (targets - y_medians) / (q3s - q1s)
        rewards = self.compute_reward(predictions, targets)  # (batch_size, num_samples)

        # log prob
        log_probs = step_log_probs  # (batch_size, num_samples, seq_len)
        # print(log_probs)

        # REINFORCE
        reinforce_loss = self.compute_reinforce_loss(log_probs, rewards)

        #
        metrics = {
            "reinforce_loss": reinforce_loss.detach(),
            "mean_reward": rewards.mean().detach(),
            "max_reward": rewards.max().detach(),
            "min_reward": rewards.min().detach(),
            "reward_std": rewards.std().detach(),
            "mean_prediction": predictions.mean().detach(),
            "prediction_std": predictions.std().detach(),
        }

        return reinforce_loss, metrics
