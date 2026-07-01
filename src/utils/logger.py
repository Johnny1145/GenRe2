import logging
from typing import Any, Dict, Optional

import wandb


class Logger:
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, name=experiment_name)

        if log_dir:
            logging.basicConfig(
                filename=f"{log_dir}/training.log",
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
            )
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Log to file
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")

    def finish(self):
        if self.use_wandb:
            wandb.finish()
