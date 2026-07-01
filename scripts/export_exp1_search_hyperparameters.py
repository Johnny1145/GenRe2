#!/usr/bin/env python3
"""Export Exp1 per-task effective hyperparameters into the release tree."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "ICLR26"
MAX_TRAIN_SIZE = 1_000_000_000
DEFAULT_CHECKPOINT_SEED = 42
DEFAULT_MERGE_CE_DIR = "results_merge_ce"
DEFAULT_DROPOUT = 0.1
CONSOLIDATED_JSON = "exp1_resolved_hyperparameters.json"
PARAM_KEYS = [
    "learning_rate",
    "base",
    "digits",
    "d_model",
    "nhead",
    "num_decoder_layers",
    "dim_feedforward",
    "hidden_dim",
    "dropout",
]
OPTUNA_PARAM_KEYS = [key for key in PARAM_KEYS if key != "dropout"]
ARCH_KEYS = ("d_model", "num_decoder_layers", "dim_feedforward", "hidden_dim")
MISMATCH_DEFAULTS = {
    "learning_rate": 1e-5,
    "base": 2,
    "digits": 8,
    "nhead": 4,
    "dropout": DEFAULT_DROPOUT,
}
OPTUNA_SOURCES = {
    "results_optuna_ce": {
        "description": (
            "Effective Exp1 parameters resolved from CE Optuna best_params and "
            "the exp1_remax_ce.py results_merge_ce checkpoint verification rule."
        ),
        "csv": "exp1_ce_optuna_best_params.csv",
        "json": CONSOLIDATED_JSON,
    },
}
_TORCH = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "PyTorch is required to infer checkpoint-backed Exp1 hyperparameters. "
                "Run this exporter in the original training environment, for example "
                "`conda run -n rlm python scripts/export_exp1_search_hyperparameters.py`."
            ) from exc
        _TORCH = torch
    return _TORCH


def discover_tasks(source_root: Path) -> list[dict[str, Any]]:
    data_root = source_root / "regression_data_new"
    if not data_root.is_dir():
        raise SystemExit(f"Missing regression_data_new directory: {data_root}")

    tasks: list[dict[str, Any]] = []
    for dataset_dir in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not dataset_dir.is_dir():
            continue
        info_path = dataset_dir / "info.json"
        if not info_path.exists():
            continue
        info = read_json(info_path)
        train_size = info.get("train_size")
        if not isinstance(train_size, (int, float)):
            continue
        if int(train_size) > MAX_TRAIN_SIZE:
            continue
        tasks.append(
            {
                "dataset": dataset_dir.name,
                "train_size": int(train_size),
                "task_type": info.get("task_type", ""),
            }
        )

    if len(tasks) != 100:
        raise SystemExit(
            f"Expected 100 filtered Exp1 tasks from {data_root}, found {len(tasks)}."
        )
    return tasks


def find_best_params(
    source_root: Path, source_name: str, dataset: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_dir = source_root / source_name
    candidates = [
        source_dir / dataset / dataset / "best_params.json",
        source_dir / dataset / "best_params.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            params = read_json(candidate)
            missing = [key for key in OPTUNA_PARAM_KEYS if key not in params]
            if missing:
                raise SystemExit(f"{candidate} is missing keys: {missing}")
            study_stats_path = candidate.with_name("study_stats.json")
            study_stats = read_json(study_stats_path) if study_stats_path.exists() else {}
            return candidate, params, study_stats
    raise SystemExit(f"Missing best_params.json for {dataset} under {source_dir}")


def resolve_merge_ce_checkpoint(
    source_root: Path,
    dataset: str,
    checkpoint_seed: int,
    merge_ce_dir: str = DEFAULT_MERGE_CE_DIR,
) -> Path | None:
    base = (
        source_root
        / merge_ce_dir
        / dataset
        / dataset
        / f"checkpoints_{checkpoint_seed}"
    )
    best_checkpoint = base / "checkpoint_best" / "model.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    if not base.exists():
        return None

    file_candidates = sorted(
        base.glob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if file_candidates:
        return file_candidates[0]

    dir_candidates = [
        path for path in base.iterdir() if path.is_dir() and path.name.startswith("checkpoint_")
    ]
    dir_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for checkpoint_dir in dir_candidates:
        for filename in ("model.pt", "checkpoint.pt"):
            candidate = checkpoint_dir / filename
            if candidate.exists():
                return candidate
    return None


def infer_model_params_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    torch = load_torch()
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    inferred: dict[str, Any] = {}

    embedding_key = "encoder_decoder.tgt_tok_emb.weight"
    if embedding_key in state_dict:
        inferred["d_model"] = state_dict[embedding_key].shape[1]

    layer_indices = set()
    layer_prefix = "encoder_decoder.decoder.layers."
    for key in state_dict:
        if layer_prefix in key:
            suffix = key.split(layer_prefix, 1)[1]
            layer_indices.add(int(suffix.split(".", 1)[0]))
    if layer_indices:
        inferred["num_decoder_layers"] = len(layer_indices)

    feedforward_key = "encoder_decoder.decoder.layers.0.linear1.weight"
    if feedforward_key in state_dict:
        inferred["dim_feedforward"] = state_dict[feedforward_key].shape[0]

    hidden_key = "encoder_decoder.custom_encoder.mlp.0.weight"
    if hidden_key in state_dict:
        inferred["hidden_dim"] = state_dict[hidden_key].shape[0]

    missing = [key for key in ARCH_KEYS if key not in inferred]
    if missing:
        raise SystemExit(
            f"{checkpoint_path} is missing checkpoint-inferred architecture keys: {missing}"
        )
    return inferred


def resolve_effective_params(
    source_root: Path,
    source_name: str,
    dataset: str,
    checkpoint_seed: int = DEFAULT_CHECKPOINT_SEED,
    merge_ce_dir: str = DEFAULT_MERGE_CE_DIR,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    optuna_params_path, optuna_params, study_stats = find_best_params(
        source_root, source_name, dataset
    )
    checkpoint_path = resolve_merge_ce_checkpoint(
        source_root, dataset, checkpoint_seed, merge_ce_dir
    )
    provenance: dict[str, Any] = {
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_relative_path": "",
        "optuna_source_relative_path": str(optuna_params_path.relative_to(source_root)),
        "architecture_mismatch_keys": "",
    }

    if checkpoint_path is None:
        effective_params = dict(optuna_params)
        effective_params.setdefault("dropout", DEFAULT_DROPOUT)
        provenance["resolution_source"] = "no_checkpoint_fallback_optuna"
    else:
        inferred = infer_model_params_from_checkpoint(checkpoint_path)
        provenance["checkpoint_relative_path"] = str(checkpoint_path.relative_to(source_root))
        mismatches = {
            key: (inferred[key], optuna_params[key])
            for key in ARCH_KEYS
            if key in inferred and key in optuna_params and inferred[key] != optuna_params[key]
        }
        if mismatches:
            effective_params = {**MISMATCH_DEFAULTS, **inferred}
            provenance["resolution_source"] = "checkpoint_defaults_due_arch_mismatch"
            provenance["architecture_mismatch_keys"] = ",".join(sorted(mismatches))
        else:
            effective_params = {**optuna_params, **inferred}
            effective_params.setdefault("dropout", DEFAULT_DROPOUT)
            provenance["resolution_source"] = "optuna_plus_checkpoint_arch_match"

    missing = [key for key in PARAM_KEYS if key not in effective_params]
    if missing:
        raise SystemExit(f"Resolved params for {dataset} are missing keys: {missing}")
    return optuna_params_path, effective_params, study_stats, provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "train_size",
        "task_type",
        *PARAM_KEYS,
        "optuna_best_value",
        "optuna_n_trials",
        "optuna_optimization_metric",
        "resolution_source",
        "checkpoint_seed",
        "checkpoint_relative_path",
        "optuna_source_relative_path",
        "architecture_mismatch_keys",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def build_resolution_rule() -> dict[str, Any]:
    return {
        "source_script": "src/RL_reweight_exp/exp1_remax_ce.py",
        "optuna_source": "results_optuna_ce",
        "checkpoint_source": DEFAULT_MERGE_CE_DIR,
        "checkpoint_seed": DEFAULT_CHECKPOINT_SEED,
        "architecture_keys": list(ARCH_KEYS),
        "mismatch_defaults": MISMATCH_DEFAULTS,
        "default_dropout": DEFAULT_DROPOUT,
    }


def resolution_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(1 for row in rows if row["resolution_source"] == key)
        for key in sorted({row["resolution_source"] for row in rows})
    }


def write_resolved_hyperparameters_json(
    path: Path, rows: list[dict[str, Any]], source_name: str
) -> None:
    payload = {
        "version": 1,
        "description": (
            "Effective hyperparameters for the Exp1 tasks that were actually run. "
            "The values follow the src/RL_reweight_exp/exp1_remax_ce.py "
            "checkpoint-verification rule."
        ),
        "source": source_name,
        "task_count": len(rows),
        "parameter_keys": PARAM_KEYS,
        "resolution_rule": build_resolution_rule(),
        "resolution_counts": resolution_counts(rows),
        "hyperparameters": {
            row["dataset"]: {key: row[key] for key in PARAM_KEYS} for row in rows
        },
        "provenance": {
            row["dataset"]: {
                "train_size": row["train_size"],
                "task_type": row["task_type"],
                "resolution_source": row["resolution_source"],
                "checkpoint_seed": row["checkpoint_seed"],
                "checkpoint_relative_path": row["checkpoint_relative_path"],
                "optuna_source_relative_path": row["optuna_source_relative_path"],
                "architecture_mismatch_keys": row["architecture_mismatch_keys"],
                "release_path": row["release_path"],
            }
            for row in rows
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_task_filtering_doc(path: Path, sources: dict[str, Any]) -> None:
    lines = [
        "# Exp1 Task Filtering and Search Hyperparameters",
        "",
        "This release includes per-task effective hyperparameters for the 100 Exp1",
        "TALENT-style regression tasks. They are resolved from CE Optuna",
        "`best_params.json` files plus the checkpoint-verification logic in the",
        "original `src/RL_reweight_exp/exp1_remax_ce.py` script.",
        "",
        "## Task Filter",
        "",
        "The 100-task set is reproduced from the later result-table scripts. The",
        "filtering rule is:",
        "",
        "1. scan `regression_data_new/*/info.json`;",
        f"2. keep datasets with numeric `train_size <= {MAX_TRAIN_SIZE}`;",
        "3. sort the dataset names lexicographically.",
        "",
        "The source implementation is the `check_train_size` and",
        "`collect_metrics_from_directories` path in the original",
        "`find_better_tasks.py` / `analyze_metrics_results_to_big_table.py` scripts.",
        "",
        "## Included Hyperparameter Sources",
        "",
    ]
    for source_name, source_meta in sources.items():
        lines.extend(
            [
                f"- `{source_name}/`: {source_meta['description']}",
                f"  Flat CSV: `docs/{source_meta['csv']}`",
                f"  Consolidated JSON: `docs/{source_meta['json']}`",
            ]
        )
    lines.extend(
        [
            "",
            "The Optuna loader first reads the nested path and then a flat fallback",
            "path:",
            "",
            "```text",
            "results_optuna_ce/<dataset>/<dataset>/best_params.json",
            "results_optuna_ce/<dataset>/best_params.json",
            "```",
            "",
            "The final released values are not a raw Optuna copy. For each dataset,",
            "the exporter resolves the checkpoint path:",
            "",
            "```text",
            f"{DEFAULT_MERGE_CE_DIR}/<dataset>/<dataset>/checkpoints_{DEFAULT_CHECKPOINT_SEED}/checkpoint_best/model.pt",
            "```",
            "",
            "It then infers `d_model`, `num_decoder_layers`, `dim_feedforward`, and",
            "`hidden_dim` from the checkpoint. If those architecture fields match",
            "Optuna, the final row uses Optuna for `learning_rate`, `base`, `digits`,",
            "and `nhead`, and uses the checkpoint-inferred architecture fields. If",
            "they do not match, the final row uses checkpoint-inferred architecture",
            "fields and the fixed defaults `learning_rate=1e-5`, `base=2`,",
            "`digits=8`, `nhead=4`, and `dropout=0.1`.",
            "",
            "`hidden_dim` is stored as a scalar in the files; training code converts",
            "it to `cfg.model.hidden_dims = [hidden_dim]`. The consolidated JSON",
            "stores `hyperparameters[dataset]` as the direct final parameter map.",
            "The CSV and JSON provenance sections also record whether a task used",
            "the `optuna_plus_checkpoint_arch_match` or",
            "`checkpoint_defaults_due_arch_mismatch` branch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_source(
    source_root: Path,
    release_root: Path,
    tasks: list[dict[str, Any]],
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_source_dir = release_root / source_name
    if output_source_dir.exists():
        shutil.rmtree(output_source_dir)

    for task in tasks:
        source_params_path, params, study_stats, provenance = resolve_effective_params(
            source_root, source_name, task["dataset"]
        )
        output_params_path = (
            output_source_dir / task["dataset"] / task["dataset"] / "best_params.json"
        )
        output_params_path.parent.mkdir(parents=True, exist_ok=True)
        with output_params_path.open("w", encoding="utf-8") as f:
            json.dump({key: params[key] for key in PARAM_KEYS}, f, indent=2)
            f.write("\n")
        rows.append(
            {
                **task,
                **{key: params[key] for key in PARAM_KEYS},
                "optuna_best_value": study_stats.get("best_value", ""),
                "optuna_n_trials": study_stats.get("n_trials", ""),
                "optuna_optimization_metric": study_stats.get("optimization_metric", ""),
                **provenance,
                "source_relative_path": str(source_params_path.relative_to(source_root)),
                "release_path": str(output_params_path.relative_to(release_root)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    docs_dir = release_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(source_root)
    manifest: dict[str, Any] = {
        "version": 1,
        "task_filter": {
            "data_dir": "regression_data_new",
            "max_train_size": MAX_TRAIN_SIZE,
            "sort": "dataset name ascending",
            "task_count": len(tasks),
        },
        "resolution_rule": build_resolution_rule(),
        "sources": {},
    }

    for source_name, source_meta in OPTUNA_SOURCES.items():
        rows = export_source(source_root, release_root, tasks, source_name)
        write_csv(docs_dir / source_meta["csv"], rows)
        write_resolved_hyperparameters_json(docs_dir / source_meta["json"], rows, source_name)
        manifest["sources"][source_name] = {
            "description": source_meta["description"],
            "csv": f"docs/{source_meta['csv']}",
            "json": f"docs/{source_meta['json']}",
            "task_count": len(rows),
            "parameters": PARAM_KEYS,
            "resolution_counts": resolution_counts(rows),
            "tasks": rows,
        }

    manifest_path = docs_dir / "exp1_search_hyperparameters_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    write_task_filtering_doc(docs_dir / "exp1_task_filtering.md", OPTUNA_SOURCES)

    print(f"Exported {len(tasks)} tasks from {source_root}")
    for source_name in OPTUNA_SOURCES:
        print(f"- {source_name}: 100 resolved best_params.json files")
    print(f"Wrote {manifest_path.relative_to(release_root)}")


if __name__ == "__main__":
    main()
