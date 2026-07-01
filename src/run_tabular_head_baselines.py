import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.run_mlp_regression_standalone import DataProcessor


class PointwiseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RiemannMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, num_bins, y_min=-3.0, y_max=3.0):
        super().__init__()
        self.num_bins = num_bins
        self.y_min = y_min
        self.y_max = y_max
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_bins + 2))
        self.net = nn.Sequential(*layers)

        width = (y_max - y_min) / num_bins
        centers = torch.linspace(y_min + 0.5 * width, y_max - 0.5 * width, num_bins)
        tail_offset = 0.5 * torch.sqrt(torch.tensor(2.0 / np.pi))
        centroids = torch.cat(
            [
                torch.tensor([y_min - tail_offset.item()]),
                centers,
                torch.tensor([y_max + tail_offset.item()]),
            ]
        )
        self.register_buffer("centroids", centroids)

    def forward(self, x):
        return self.net(x)

    def expected_value(self, x):
        probs = torch.softmax(self.forward(x), dim=-1)
        return torch.sum(probs * self.centroids.to(probs.device), dim=-1)


def make_histogram_targets(y, num_bins, y_min=-3.0, y_max=3.0):
    y = y.view(-1)
    width = (y_max - y_min) / num_bins
    centers = torch.linspace(y_min + 0.5 * width, y_max - 0.5 * width, num_bins, device=y.device)
    sigma = 0.75 * width
    central = torch.exp(-0.5 * ((y[:, None] - centers[None, :]) / sigma) ** 2)
    central = central / central.sum(dim=1, keepdim=True).clamp_min(1e-12)

    targets = torch.zeros((y.shape[0], num_bins + 2), device=y.device)
    left = y < y_min
    right = y > y_max
    middle = ~(left | right)
    targets[left, 0] = 1.0
    targets[right, -1] = 1.0
    targets[middle, 1:-1] = central[middle]
    return targets


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "rank_correlation": float(spearmanr(y_true, y_pred).correlation),
    }


def make_loader(x, y, batch_size, shuffle):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def train_one_dataset(args, dataset_name, data_processor, device):
    data = data_processor.load_dataset(dataset_name)
    x_train = data["N"]["train"]
    x_val = data["N"]["val"]
    x_test = data["N"]["test"]
    y_train = data["y"]["train"].reshape(-1)
    y_val = data["y"]["val"].reshape(-1)

    hidden_dims = [args.hidden_dim] * args.num_layers
    if args.head == "pointwise":
        model = PointwiseMLP(x_train.shape[1], hidden_dims, args.dropout).to(device)
    else:
        model = RiemannMLP(x_train.shape[1], hidden_dims, args.dropout, args.num_bins).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(x_train, y_train, args.batch_size, True)
    val_loader = make_loader(x_val, y_val, args.batch_size, False)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps - 1) if total_steps > 1 else 0
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_steps - warmup_steps),
                    eta_min=args.learning_rate * args.min_lr_ratio,
                ),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = None

    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            if args.head == "pointwise":
                loss = F.mse_loss(model(batch_x), batch_y)
            else:
                logits = model(batch_x)
                soft_targets = make_histogram_targets(batch_y, args.num_bins)
                loss = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if args.head == "pointwise":
                    pred = model(batch_x)
                    val_loss = F.mse_loss(pred, batch_y)
                else:
                    pred = model.expected_value(batch_x)
                    val_loss = F.mse_loss(pred, batch_y)
                val_losses.append(val_loss.item())
        val_mean = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = make_loader(x_test, data["y"]["test"].reshape(-1), args.batch_size, False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            if args.head == "pointwise":
                pred = model(batch_x)
            else:
                pred = model.expected_value(batch_x)
            predictions.append(pred.cpu().numpy())

    y_info = data["y_info"]
    y_true = data["y"]["test"].reshape(-1)
    y_true = (y_true - y_info["mean"]) / y_info["std"]
    y_pred = np.concatenate(predictions).reshape(-1)
    metrics = compute_metrics(y_true, y_pred)
    return {
        "dataset_name": dataset_name,
        "head": args.head,
        "best_val_loss": float(best_val),
        "metrics": metrics,
        "predictions": [float(x) for x in y_pred.tolist()],
        "targets": [float(x) for x in y_true.tolist()],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train pointwise or Riemann tabular head baselines.")
    parser.add_argument("--head", choices=["pointwise", "riemann"], default="pointwise")
    parser.add_argument("--data-dir", default="data/talent")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--results-dir", default="outputs/tabular_heads")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--num-bins", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_processor = DataProcessor(args.data_dir)
    datasets = [args.dataset] if args.dataset else data_processor.get_all_datasets()
    failures = []
    for dataset_name in tqdm(datasets, desc="datasets"):
        try:
            result = train_one_dataset(args, dataset_name, data_processor, device)
            result_dir = os.path.join(args.results_dir, dataset_name)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as exc:
            failures.append({"dataset_name": dataset_name, "error": str(exc)})
    if failures:
        with open(os.path.join(args.results_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)


if __name__ == "__main__":
    main()
