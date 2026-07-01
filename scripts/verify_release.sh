#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "== Shell syntax =="
find scripts -type f -name '*.sh' -print0 | sort -z | xargs -0 -n1 bash -n

echo "== Release metadata =="
test -f LICENSE
test -f README.md
test -f docs/exp1_model_architectures.md
test -f docs/exp1_task_filtering.md
test -f docs/exp1_ce_optuna_best_params.csv
test -f docs/exp1_resolved_hyperparameters.json
test -f docs/exp1_search_hyperparameters_manifest.json
python - <<'PY'
from pathlib import Path
import csv
import json

readme = Path("README.md").read_text(encoding="utf-8")
exp1 = Path("docs/exp1_model_architectures.md").read_text(encoding="utf-8")
task_filtering = Path("docs/exp1_task_filtering.md").read_text(encoding="utf-8")

required_methods = [
    "Pointwise head baseline",
    "Riemann head baseline",
    "CE hyperparameter search",
    "CE",
    "NTL-MSE",
    "NTL-WAS",
    "DIST2",
    "ReMax",
    "GenRe2",
]
missing = [method for method in required_methods if method not in exp1]
if missing:
    raise SystemExit(f"Exp1 architecture doc is missing: {missing}")

required_wrappers = [
    "scripts/tabular_pointwise.sh",
    "scripts/tabular_riemann.sh",
    "scripts/tabular_search_ce.sh",
    "scripts/tabular_ce.sh",
    "scripts/tabular_ntl_mse.sh",
    "scripts/tabular_ntl_was.sh",
    "scripts/tabular_dist2.sh",
    "scripts/tabular_remax.sh",
    "scripts/tabular_genre2.sh",
]
missing_exp1_wrappers = [wrapper for wrapper in required_wrappers if wrapper not in exp1]
if missing_exp1_wrappers:
    raise SystemExit(f"Missing Exp1 wrapper references: {missing_exp1_wrappers}")

required_readme_entries = [
    "## Environment installation",
    "## Main experiments",
    "conda env create -f environment.yml",
    "pip install -r requirements.txt",
    "accelerate config",
    "TABULAR_DATA_DIR=/path/to/talent",
    "scripts/tabular_search_ce.sh",
    "scripts/tabular_genre2.sh",
    "RLM_DATA_DIR=/path/to/code_metric",
    "scripts/rlm_genre2.sh",
    "GRM_MODEL=/path/to/sft_model",
    "scripts/grm_genre2.sh",
    "scripts/grm_eval.sh",
]
missing_readme_entries = [entry for entry in required_readme_entries if entry not in readme]
if missing_readme_entries:
    raise SystemExit(f"README is missing experiment/setup entries: {missing_readme_entries}")

if "train_size <= 1000000000" not in task_filtering:
    raise SystemExit("Task filtering doc does not record the 100-task train_size rule.")

if "OPD_WEIGHT` / `reinforce.expert_ce_weight` | `0.05`" not in exp1:
    raise SystemExit("Exp1 architecture doc does not record OPD_WEIGHT=0.05.")

param_keys = [
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

manifest = json.loads(Path("docs/exp1_search_hyperparameters_manifest.json").read_text(encoding="utf-8"))
if manifest["task_filter"]["task_count"] != 100:
    raise SystemExit("Search hyperparameter manifest does not record exactly 100 tasks.")
if set(manifest["sources"]) != {"results_optuna_ce"}:
    raise SystemExit(f"Unexpected hyperparameter sources in manifest: {sorted(manifest['sources'])}")
if manifest["resolution_rule"]["source_script"] != "src/RL_reweight_exp/exp1_remax_ce.py":
    raise SystemExit("Manifest does not record the exp1_remax_ce.py resolution rule.")
if manifest["resolution_rule"]["checkpoint_seed"] != 42:
    raise SystemExit("Manifest does not record checkpoint_seed=42.")

source = manifest["sources"]["results_optuna_ce"]
if source["task_count"] != 100:
    raise SystemExit("results_optuna_ce manifest source does not contain exactly 100 tasks.")
if source["parameters"] != param_keys:
    raise SystemExit("Manifest parameter key order/content changed unexpectedly.")
if source["json"] != "docs/exp1_resolved_hyperparameters.json":
    raise SystemExit("Manifest does not point to the consolidated resolved hyperparameter JSON.")
expected_resolution_counts = {
    "checkpoint_defaults_due_arch_mismatch": 41,
    "optuna_plus_checkpoint_arch_match": 59,
}
if source["resolution_counts"] != expected_resolution_counts:
    raise SystemExit(
        f"Unexpected hyperparameter resolution counts: {source['resolution_counts']}"
    )

with Path("docs/exp1_ce_optuna_best_params.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 100:
    raise SystemExit(f"Expected 100 CSV hyperparameter rows, found {len(rows)}.")

resolved_json = json.loads(Path("docs/exp1_resolved_hyperparameters.json").read_text(encoding="utf-8"))
if resolved_json["task_count"] != 100:
    raise SystemExit("Resolved hyperparameter JSON does not contain exactly 100 tasks.")
if resolved_json["parameter_keys"] != param_keys:
    raise SystemExit("Resolved hyperparameter JSON parameter key order/content changed unexpectedly.")
if resolved_json["resolution_counts"] != expected_resolution_counts:
    raise SystemExit(
        f"Resolved hyperparameter JSON has unexpected resolution counts: {resolved_json['resolution_counts']}"
    )

csv_resolution_counts = {}
for row in rows:
    csv_resolution_counts[row["resolution_source"]] = csv_resolution_counts.get(row["resolution_source"], 0) + 1
    if row["resolution_source"] == "checkpoint_defaults_due_arch_mismatch":
        expected_defaults = {
            "learning_rate": "1e-05",
            "base": "2",
            "digits": "8",
            "nhead": "4",
            "dropout": "0.1",
        }
        for key, expected in expected_defaults.items():
            if row[key] != expected:
                raise SystemExit(
                    f"Mismatch branch row {row['dataset']} has {key}={row[key]}, expected {expected}."
                )
if csv_resolution_counts != expected_resolution_counts:
    raise SystemExit(f"CSV resolution counts do not match manifest: {csv_resolution_counts}")

manifest_tasks = {task["dataset"] for task in source["tasks"]}
csv_tasks = {row["dataset"] for row in rows}
if manifest_tasks != csv_tasks:
    raise SystemExit("CSV task set does not match manifest task set.")
json_tasks_from_consolidated = set(resolved_json["hyperparameters"])
if json_tasks_from_consolidated != manifest_tasks:
    raise SystemExit("Resolved hyperparameter JSON task set does not match manifest task set.")

json_paths = sorted(Path("results_optuna_ce").glob("*/*/best_params.json"))
if len(json_paths) != 100:
    raise SystemExit(f"Expected 100 results_optuna_ce best_params.json files, found {len(json_paths)}.")
json_tasks = {path.parts[-3] for path in json_paths}
if json_tasks != manifest_tasks:
    raise SystemExit("results_optuna_ce task set does not match manifest task set.")

csv_rows_by_task = {row["dataset"]: row for row in rows}
manifest_rows_by_task = {task["dataset"]: task for task in source["tasks"]}
for path in json_paths:
    params = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in param_keys if key not in params]
    if missing:
        raise SystemExit(f"{path} is missing searched hyperparameter keys: {missing}")
    dataset = path.parts[-3]
    csv_row = csv_rows_by_task[dataset]
    manifest_row = manifest_rows_by_task[dataset]
    consolidated_params = resolved_json["hyperparameters"][dataset]
    for key in param_keys:
        if str(params[key]) != csv_row[key]:
            raise SystemExit(f"{path} {key} does not match CSV value for {dataset}.")
        if params[key] != manifest_row[key]:
            raise SystemExit(f"{path} {key} does not match manifest value for {dataset}.")
        if params[key] != consolidated_params[key]:
            raise SystemExit(f"{path} {key} does not match resolved JSON value for {dataset}.")

for train_file in Path("src/search").glob("train_search_*.py"):
    text = train_file.read_text(encoding="utf-8")
    if "results_optuna_ce_new" in text:
        raise SystemExit(f"{train_file} still defaults to results_optuna_ce_new.")

for stale in [
    "upload_selected_weights.sh",
    "upload_rlm_table2_weights.sh",
    "generate_weight_manifest.py",
    "model_cards/",
]:
    if stale in readme:
        raise SystemExit(f"README still references weight release artifact: {stale}")

print("Validated README, Exp1 docs, resolved hyperparameters, and LICENSE.")
PY

echo "== Secret scan =="
secret_hits="$(rg -n 'mifJ3wKj|api_key=\"[A-Za-z0-9_/-]{12,}\"|hf_[A-Za-z0-9]{20,}' . \
  --glob '!docs/**' \
  --glob '!scripts/verify_release.sh' \
  --glob '!*.tar.gz' || true)"
if [[ -n "${secret_hits}" ]]; then
  echo "${secret_hits}"
  echo "Potential hard-coded secret found." >&2
  exit 1
fi
echo "No hard-coded secret patterns found."

echo "== CJK character scan =="
cjk_hits="$(rg -n '[\p{Han}]' README.md docs scripts src requirements.txt environment.yml LICENSE || true)"
if [[ -n "${cjk_hits}" ]]; then
  echo "${cjk_hits}"
  echo "CJK characters should not be included in release code, scripts, or docs." >&2
  exit 1
fi
echo "No CJK characters found."

echo "== Local path scan =="
local_path_hits="$(rg -n '/home/trx|/mnt/|/private/|/Users/' . \
  --glob '!scripts/verify_release.sh' \
  --glob '!*.tar.gz' || true)"
if [[ -n "${local_path_hits}" ]]; then
  echo "${local_path_hits}"
  echo "Local absolute paths should not be included in the code release." >&2
  exit 1
fi
echo "No local absolute paths found."

echo "== Wrapper dry-runs =="
for wrapper in \
  scripts/tabular_pointwise.sh \
  scripts/tabular_riemann.sh \
  scripts/tabular_search_ce.sh \
  scripts/tabular_ce.sh \
  scripts/tabular_ntl_mse.sh \
  scripts/tabular_ntl_was.sh \
  scripts/tabular_dist2.sh \
  scripts/tabular_remax.sh \
  scripts/tabular_genre2.sh; do
  DRY_RUN=1 GPUS=0 NUM_PROCESSES=1 TABULAR_DATA_DIR=/tmp/talent bash "${wrapper}" Abalone_reg
done

for wrapper in \
  scripts/rlm_base.sh \
  scripts/rlm_ce.sh \
  scripts/rlm_ntl_mse.sh \
  scripts/rlm_ntl_was.sh \
  scripts/rlm_dist2.sh \
  scripts/rlm_remax.sh \
  scripts/rlm_genre2.sh; do
  DRY_RUN=1 GPUS=0 NUM_PROCESSES=1 RLM_DATA_DIR=/tmp/code_metric bash "${wrapper}" apps
done

for wrapper in \
  scripts/grm_sft.sh \
  scripts/grm_dist2.sh \
  scripts/grm_remax.sh \
  scripts/grm_genre2.sh; do
  DRY_RUN=1 GPUS=0 NUM_PROCESSES=1 GRM_MODEL=/tmp/sft bash "${wrapper}"
done

DRY_RUN=1 GRM_EVAL_MODEL=/tmp/sft GRM_EVAL_NUM_GPUS=1 bash scripts/grm_eval.sh --debug

echo "== No weight publication artifacts =="
stale_weight_files="$(find scripts model_cards dataset_cards docs -type f 2>/dev/null | rg 'upload_.*weight|upload_selected_weights|generate_weight_manifest|weight_file_manifest|weight_release_manifest|model_cards/.+/README|dataset_cards/.+/README' || true)"
if [[ -n "${stale_weight_files}" ]]; then
  echo "${stale_weight_files}"
  echo "Weight publication artifacts should not be included in this code/hyperparameter release." >&2
  exit 1
fi
echo "No weight publication artifacts found."

echo "== No Python cache artifacts =="
cache_artifacts="$(find . \( -type d -name '__pycache__' -o -type f -name '*.pyc' \) -not -path './.git/*' -print)"
if [[ -n "${cache_artifacts}" ]]; then
  echo "${cache_artifacts}"
  echo "Python cache artifacts should not be included in the code release." >&2
  exit 1
fi
echo "No Python cache artifacts found."

echo "== No generated training outputs =="
generated_outputs="$(find . -maxdepth 1 -type d \( \
  -name 'outputs' -o \
  -name 'logs' -o \
  -name 'wandb' -o \
  -name 'swanlog' -o \
  -name 'results_search_mlp_encoder*' -o \
  -name 'results_exp*' \
\) -print)"
if [[ -n "${generated_outputs}" ]]; then
  echo "${generated_outputs}"
  echo "Generated training outputs should not be included in the code release." >&2
  exit 1
fi
echo "No generated training outputs found."

echo "== Large files in code release =="
large_files="$(find . -type f -size +10M -not -path './.git/*' -print)"
if [[ -n "${large_files}" ]]; then
  echo "${large_files}"
  echo "Unexpected large files found in the code release." >&2
  exit 1
fi
echo "No files larger than 10MB found."

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
  echo "== Runtime smoke tests =="
  PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test_release.py
else
  echo "== Runtime smoke tests =="
  echo "Skipped. Set RUN_SMOKE=1 in an environment with release dependencies installed."
fi

echo "Release checks completed."
