#!/usr/bin/env python
"""Lightweight runtime smoke tests for the code/architecture release."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def run(cmd: list[str], *, timeout: int = 120) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=timeout)


def make_talent_dataset(root: Path) -> Path:
    data_dir = root / "talent"
    dataset_dir = data_dir / "smoke_reg"
    dataset_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(12, 4)).astype("float32")
    x_val = rng.normal(size=(6, 4)).astype("float32")
    x_test = rng.normal(size=(6, 4)).astype("float32")

    def target(x: np.ndarray) -> np.ndarray:
        return (0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.1).astype("float32")

    arrays = {
        "N_train.npy": x_train,
        "N_val.npy": x_val,
        "N_test.npy": x_test,
        "y_train.npy": target(x_train),
        "y_val.npy": target(x_val),
        "y_test.npy": target(x_test),
    }
    for name, value in arrays.items():
        np.save(dataset_dir / name, value)

    info = {
        "name": "smoke_reg",
        "task_type": "regression",
        "n_num_features": 4,
        "n_cat_features": 0,
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
    }
    (dataset_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return data_dir


def import_smoke() -> None:
    modules = [
        "src.run_tabular_head_baselines",
        "src.search.search_ce",
        "src.search.train_search_ce",
        "src.search.train_search_ntl_mse",
        "src.search.train_search_ntl_was",
        "src.search.train_search_dist2",
        "src.search.train_search_rl",
        "src.search.train_search_rl_expert",
        "src.rlm_exp.zero_shot",
        "src.rlm_exp.train_ce",
        "src.rlm_exp.train_ntl_new",
        "src.rlm_exp.train_ntl_was",
        "src.rlm_exp.train_DIST2",
        "src.rlm_exp.train_rl_new",
        "src.rlm_exp.train_rl_expert",
        "src.generative_reward_models.trainer.sft_Mistral",
        "src.generative_reward_models.trainer.Dist2_Mistral",
        "src.generative_reward_models.trainer.ReMax_Mistral",
        "src.generative_reward_models.trainer.ReMax_expert_Mistral",
        "src.generative_reward_models.evaluation.reward_bench_generative",
        "src.generative_reward_models.evaluation.reward_bench_linear",
    ]
    for module in modules:
        print(f"import {module}", flush=True)
        __import__(module)


def tabular_head_smoke(data_dir: Path, out_dir: Path) -> None:
    common = [
        sys.executable,
        "-m",
        "src.run_tabular_head_baselines",
        "--data-dir",
        str(data_dir),
        "--dataset",
        "smoke_reg",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--hidden-dim",
        "8",
        "--warmup-steps",
        "0",
    ]
    run(common + ["--head", "pointwise", "--results-dir", str(out_dir / "pointwise")])
    run(
        common
        + [
            "--head",
            "riemann",
            "--num-layers",
            "1",
            "--num-bins",
            "8",
            "--results-dir",
            str(out_dir / "riemann"),
        ]
    )
    for head in ["pointwise", "riemann"]:
        result = out_dir / head / "smoke_reg" / "results.json"
        if not result.exists():
            raise RuntimeError(f"missing smoke result: {result}")


def grm_eval_cli_smoke() -> None:
    run([sys.executable, "-m", "src.generative_reward_models.evaluation.reward_bench_generative", "--help"])
    run(["bash", "scripts/grm_eval.sh", "--help"])


def main() -> None:
    import_smoke()
    with tempfile.TemporaryDirectory(prefix="genre2-release-smoke-") as tmp:
        tmp_path = Path(tmp)
        data_dir = make_talent_dataset(tmp_path)
        tabular_head_smoke(data_dir, tmp_path / "outputs")
    grm_eval_cli_smoke()
    print("Smoke tests completed.")


if __name__ == "__main__":
    main()
