import json
import os
from typing import Dict, Optional

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_metrics = {}
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, module, epoch: int | str, metrics: Dict):
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}")
        # module.accelerator.save_state(checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)
        unwrapped_model = module.accelerator.unwrap_model(module.model)
        torch.save(
            unwrapped_model.state_dict(), os.path.join(checkpoint_path, "model.pt")
        )

        # Save metrics
        metrics_path = os.path.join(checkpoint_path, "metrics.json")
        # TensorPythonJSON
        serializable_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                serializable_metrics[key] = (
                    value.item() if value.numel() == 1 else value.tolist()
                )
            else:
                serializable_metrics[key] = value

        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f)

        # Update best metrics (only for epoch-based checkpoints, not step-based)
        if isinstance(epoch, int):
            for metric_name, value in metrics.items():
                if (
                    metric_name not in self.best_metrics
                    or value < self.best_metrics[metric_name]["value"]
                ):
                    self.best_metrics[metric_name] = {
                        "value": value,
                        "epoch": epoch,
                        "path": checkpoint_path,
                    }

    def load_best_checkpoint(self, module, metric_name: str):
        if metric_name in self.best_metrics:
            best_checkpoint_path = self.best_metrics[metric_name]["path"]
            module.accelerator.load_state(best_checkpoint_path)
            return True
        return False
