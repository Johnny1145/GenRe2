import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd

class CustomMLP(nn.Module):
    """

    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.1):
        super(CustomMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        #
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        #
        layers.append(nn.Linear(prev_dim, 1))

        #
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DataProcessor:
    """

    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.y_info = {}
        self.X_info = {}

    def load_dataset(self, dataset_name):
        """
        numeric_regression_with_y_norm_dataset
        -
        - Ordinal//
        - StandardScaler
        -
        - /train/valtest
        """
        dataset_path = os.path.join(self.data_dir, dataset_name)

        # info.json
        info_path = os.path.join(dataset_path, 'info.json')
        info = None
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                try:
                    info = json.load(f)
                except Exception:
                    info = None

        #
        def _safe_load(path):
            return np.load(path, allow_pickle=True) if os.path.exists(path) else None

        N_train = _safe_load(os.path.join(dataset_path, 'N_train.npy'))
        N_val = _safe_load(os.path.join(dataset_path, 'N_val.npy'))
        N_test = _safe_load(os.path.join(dataset_path, 'N_test.npy'))

        C_train = _safe_load(os.path.join(dataset_path, 'C_train.npy'))
        C_val = _safe_load(os.path.join(dataset_path, 'C_val.npy'))
        C_test = _safe_load(os.path.join(dataset_path, 'C_test.npy'))

        # 1
        def _ensure_2d(x):
            if x is None:
                return None
            if len(x.shape) == 1:
                return x.reshape(-1, 1)
            return x

        N_train, N_val, N_test = map(_ensure_2d, (N_train, N_val, N_test))
        C_train, C_val, C_test = map(_ensure_2d, (C_train, C_val, C_test))

        #
        if N_train is not None:
            num_mean = np.nanmean(N_train.astype(float), axis=0)
            num_mean = np.nan_to_num(num_mean)
            def _fill_num(x):
                if x is None:
                    return None
                x = x.astype(float)
                nan_mask = np.isnan(x)
                if nan_mask.any():
                    inds = np.where(nan_mask)
                    x[inds] = np.take(num_mean, inds[1])
                return x
            N_train = _fill_num(N_train)
            N_val = _fill_num(N_val)
            N_test = _fill_num(N_test)

        if C_train is not None:
            #
            C_train = C_train.astype(str)
            C_val = C_val.astype(str) if C_val is not None else None
            C_test = C_test.astype(str) if C_test is not None else None

            #
            def _fill_cat(x):
                if x is None:
                    return None
                mask = np.isin(x, ["nan", "NaN", "", None])
                if mask.any():
                    x[mask] = "___null___"
                return x
            C_train = _fill_cat(C_train)
            C_val = _fill_cat(C_val)
            C_test = _fill_cat(C_test)

            # Ordinal val/testtrain
            unknown_value = np.iinfo('int64').max - 3
            ord_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=unknown_value,
                dtype='int64'
            )
            ord_encoder.fit(C_train)
            C_train_enc = ord_encoder.transform(C_train)
            C_val_enc = ord_encoder.transform(C_val) if C_val is not None else None
            C_test_enc = ord_encoder.transform(C_test) if C_test is not None else None

            # modeunknown_value
            mode_values = None
            if C_val_enc is not None or C_test_enc is not None:
                mode_values = []
                for col in range(C_train_enc.shape[1]):
                    column = C_train_enc[:, col]
                    valid = column[column != unknown_value]
                    if valid.size == 0:
                        mode_values.append(0)
                    else:
                        counts = np.bincount(valid)
                        mode_values.append(np.argmax(counts))
                mode_values = np.array(mode_values)

            def _replace_unknown(x):
                if x is None:
                    return None
                for col in range(x.shape[1]):
                    mask = x[:, col] == unknown_value
                    if mask.any():
                        x[mask, col] = mode_values[col]
                return x

            C_val_enc = _replace_unknown(C_val_enc)
            C_test_enc = _replace_unknown(C_test_enc)
        else:
            C_train_enc = C_val_enc = C_test_enc = None

        # StandardScaler,
        if N_train is not None:
            scaler = StandardScaler().fit(N_train)
            N_train = scaler.transform(N_train)
            N_val = scaler.transform(N_val) if N_val is not None else None
            N_test = scaler.transform(N_test) if N_test is not None else None

        #
        def _concat(a, b):
            if a is not None and b is not None:
                return np.concatenate([a, b], axis=1)
            return a if b is None else b

        X_train = _concat(N_train, C_train_enc)
        X_val = _concat(N_val, C_val_enc)
        X_test = _concat(N_test, C_test_enc)

        #
        y_train = np.load(os.path.join(dataset_path, 'y_train.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(dataset_path, 'y_val.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(dataset_path, 'y_test.npy'), allow_pickle=True)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)

        # y
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train)) if float(np.std(y_train)) > 0 else 1.0
        self.y_info[dataset_name] = { 'mean': y_mean, 'std': y_std }

        # train/valtest/
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std

        # 'N'
        return {
            'N': {'train': X_train, 'val': X_val, 'test': X_test},
            'y': {'train': y_train_norm, 'val': y_val_norm, 'test': y_test},
            'y_info': self.y_info[dataset_name]
        }

    def get_all_datasets(self):
        """
        info.json
        """
        datasets = []
        for dataset_name in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            info_file = os.path.join(dataset_path, 'info.json')
            if not os.path.exists(info_file):
                continue
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                if info.get('task_type') != 'regression':
                    continue
            except Exception:
                pass
            # y
            has_any_feature = (
                os.path.exists(os.path.join(dataset_path, 'N_train.npy')) or
                os.path.exists(os.path.join(dataset_path, 'C_train.npy'))
            )
            has_y = os.path.exists(os.path.join(dataset_path, 'y_train.npy'))
            if has_any_feature and has_y:
                datasets.append(dataset_name)
        return datasets

class ModelTrainer:
    """

    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3,
                 lr=0.0003, warmup_steps=100, min_lr_ratio=0.1):
        # GPUCPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        self.model = CustomMLP(input_dim, hidden_dims, dropout).to(self.device)
        #
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.scheduler = None

    def train(self, train_data, val_data, batch_size=128, epochs=50):
        """

        """
        #
        train_loader = self._prepare_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(val_data, batch_size, shuffle=False)

        #
        self._setup_scheduler(epochs, len(train_loader))

        best_val_loss = float('inf')
        best_model_state = None

        #
        for epoch in range(epochs):
            #
            self.model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                #
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets.squeeze())

                #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #
                self.scheduler.step()

                train_loss += loss.item() * inputs.size(0)

            #
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(), targets.squeeze())
                    val_loss += loss.item() * inputs.size(0)

            #
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)

            #
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {current_lr:.6f}")

            #
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()

        #
        self.model.load_state_dict(best_model_state)
        return best_val_loss

    def _setup_scheduler(self, epochs, steps_per_epoch):
        """
        warmup + cosine annealing
        """
        total_steps = epochs * steps_per_epoch

        # Warmup0
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )

        # Cosine annealingmin_lr_ratio * initial_lr
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
            eta_min=self.optimizer.param_groups[0]['lr'] * self.min_lr_ratio
        )

        #
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

    def predict(self, test_data):
        """

        """
        #
        test_loader = self._prepare_dataloader(test_data, batch_size=128, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    def _prepare_dataloader(self, data, batch_size, shuffle):
        """

        """
        inputs = torch.tensor(data['X'], dtype=torch.float32)
        targets = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class MetricsCalculator:
    """

    """
    @staticmethod
    def compute_metrics(y_true, y_pred):
        """

        """

        #
        mse_value = mean_squared_error(y_true, y_pred)
        mae_value = mean_absolute_error(y_true, y_pred)
        rmse_value = np.sqrt(mse_value)
        r2_value = r2_score(y_true, y_pred)
        spearman_value = spearmanr(y_true, y_pred).correlation

        return {
            'mse': mse_value,
            'mae': mae_value,
            'rmse': rmse_value,
            'r2': r2_value,
            'rank_correlation': spearman_value
        }

def run_mlp_regression_standalone(
    data_dir='data/talent',
    results_dir='outputs/pointwise_mlp',
    epochs=100,
    batch_size=16,
    hidden_dim=2048,
    learning_rate=1e-5,
):
    os.makedirs(results_dir, exist_ok=True)

    #
    data_processor = DataProcessor(data_dir)

    #
    datasets = data_processor.get_all_datasets()
    print(f" {len(datasets)} ")

    #
    all_results = []
    failed_datasets = []

    #
    for dataset_name in tqdm(datasets, desc=''):
        try:
            print(f": {dataset_name}")

            #
            data = data_processor.load_dataset(dataset_name)

            # N
            X_train = data['N']['train']
            X_val = data['N']['val']
            X_test = data['N']['test']

            #
            train_data = {'X': X_train, 'y': data['y']['train']}
            val_data = {'X': X_val, 'y': data['y']['val']}
            test_data = {'X': X_test, 'y': data['y']['test']}  # y_test

            #
            input_dim = X_train.shape[1] if X_train is not None else 0
            trainer = ModelTrainer(
                input_dim,
                hidden_dims=[hidden_dim],
                dropout=0.1,
                lr=learning_rate,
                warmup_steps=100,
                min_lr_ratio=0.1
            )

            #
            best_val_loss = trainer.train(train_data, val_data, batch_size=batch_size, epochs=epochs)
            print(f": {best_val_loss:.4f}")

            #
            y_info = data['y_info']
            predictions = trainer.predict(test_data).flatten()
            targets = data['y']['test'].flatten()
            targets = (targets -y_info['mean'])/y_info['std']

            # NumPyPythonPython
            predictions = predictions.tolist()
            targets = targets.tolist()

            # Pythonfloat
            predictions = [float(pred) for pred in predictions]
            targets = [float(target) for target in targets]

            #
            metrics = MetricsCalculator.compute_metrics(targets, predictions)
            # Pythonfloat
            metrics = {key: float(value) for key, value in metrics.items()}
            print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

            #
            result = {
                'dataset_name': dataset_name,
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets  # targets
            }

            #
            dataset_result_dir = os.path.join(results_dir, dataset_name)
            os.makedirs(dataset_result_dir, exist_ok=True)

            # JSON
            with open(os.path.join(dataset_result_dir, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)


        except Exception as e:
            error_msg = f" {dataset_name} : {str(e)}"
            print(error_msg)
            failed_datasets.append({'Dataset': dataset_name, 'Reason': error_msg})
            continue

    #
    if failed_datasets:
        print(f": {len(failed_datasets)}")
        for item in failed_datasets:
            print(f"- {item['Dataset']}: {item['Reason']}")

    print("")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the pointwise MLP tabular baseline.")
    parser.add_argument("--data-dir", default="data/talent")
    parser.add_argument("--results-dir", default="outputs/pointwise_mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_mlp_regression_standalone(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )
