# GenRe2 Main Experiments

This directory is a cleaned release copy for the main experiments of the GenRe2 paper. It includes only the code needed for:

- Tabular regression on TALENT-style datasets.
- RLM code metric regression.
- Generative reward model training and evaluation.
- Exp1 tabular model architecture parameter disclosure.
- Exp1 per-task resolved CE hyperparameters for the 100 filtered tasks.

Appendix-only ablations, plotting scripts, temporary tests, cached files, generated figures, and result dumps are intentionally excluded.
Model weights are intentionally not part of this release scope.

## Layout

```text
src/
  run_tabular_head_baselines.py # Pointwise and Riemann tabular baselines
  search/search_ce.py           # Optuna search for the tabular CE base model
  search/                       # Tabular CE, token-level baselines, ReMax, GenRe2
  rlm_exp/                      # Code metric regression experiments
  generative_reward_models/     # GRM SFT, DIST2, ReMax, GenRe2, and evaluation utilities
  model/                        # RegressLM model code
  data/                         # Dataset loaders
  utils/                        # Losses, RL objectives, and checkpoint helpers
scripts/
  run_main_experiments.sh       # Shared implementation used by the wrappers below
  smoke_test_release.py          # Lightweight import/CLI/runtime smoke tests
results_optuna_ce/
  <dataset>/<dataset>/best_params.json # 100 task resolved Exp1 params
docs/exp1_resolved_hyperparameters.json # consolidated final task hyperparameters
```

## Environment

Create a fresh environment:

```bash
conda env create -f environment.yml
conda activate genre2-main
```

Or install the release dependencies into an existing environment:

```bash
conda activate rlm
pip install -r requirements.txt
```

The training scripts use `accelerate` for multi-GPU runs. Configure it once if needed:

```bash
accelerate config
```

For quick command validation without launching training, set `DRY_RUN=1`.
For small smoke tests, set `MAX_ITEMS` where supported.

## License

The code release is distributed under the Apache License 2.0; see `LICENSE`.
The published GRM model cards use Apache-2.0 metadata. The partial RLM model card uses MIT metadata to match the upstream public base checkpoint `akhauriyash/RLM-GemmaS-Code-v0`.
Check third-party dataset and model terms before redistributing derived artifacts.

## Data

The code expects data under relative paths by default:

- `data/talent` for TALENT-style tabular regression data.
- `data/code_metric` for RLM code metric JSONL data.

Override these paths with:

```bash
export TABULAR_DATA_DIR=/path/to/talent
export RLM_DATA_DIR=/path/to/code_metric
```

GRM experiments also need model paths:

```bash
export GRM_MODEL=/path/to/sft_or_base_model
export GRM_REF_MODEL=prometheus-eval/prometheus-8x7b-v2.0
```

## Tabular Scripts

Exp1 architecture parameters for each tabular model are documented in:

```text
docs/exp1_model_architectures.md
```

The tabular CE/base decoder hyperparameters are searched with Optuna. Run that search before reusing the selected settings for CE, token-level baselines, ReMax, or GenRe2:

```bash
TABULAR_DATA_DIR=/path/to/talent N_TRIALS=25 bash scripts/tabular_search_ce.sh Abalone_reg
```

This release also includes the resolved Exp1 CE hyperparameters for the 100
filtered tasks:

```text
results_optuna_ce/<dataset>/<dataset>/best_params.json
docs/exp1_resolved_hyperparameters.json
docs/exp1_ce_optuna_best_params.csv
docs/exp1_search_hyperparameters_manifest.json
docs/exp1_task_filtering.md
```

The 100-task rule is documented in `docs/exp1_task_filtering.md`: scan
`regression_data_new/*/info.json`, keep datasets with numeric
`train_size <= 1000000000`, then sort dataset names lexicographically.
The released values follow the original `src/RL_reweight_exp/exp1_remax_ce.py`
logic: read CE Optuna `best_params.json`, infer architecture fields from the
`results_merge_ce` checkpoint, and use fixed defaults when the checkpoint
architecture does not match Optuna.

Standalone tabular scripts:

```text
scripts/tabular_pointwise.sh
scripts/tabular_riemann.sh
scripts/tabular_search_ce.sh
scripts/tabular_ce.sh
scripts/tabular_ntl_mse.sh
scripts/tabular_ntl_was.sh
scripts/tabular_dist2.sh
scripts/tabular_remax.sh
scripts/tabular_genre2.sh
```

Example:

```bash
GPUS=0 NUM_PROCESSES=1 TABULAR_DATA_DIR=/path/to/talent \
  bash scripts/tabular_genre2.sh Abalone_reg
```

Tabular NTL, DIST2, ReMax, and GenRe2 default to initializing from the CE
checkpoint produced by `scripts/tabular_ce.sh`:

```text
results_search_mlp_encoder_ce/<dataset>/<dataset>/checkpoints_<seed>/checkpoint_best/model.pt
```

Override this with a Hydra argument when using a different checkpoint:

```bash
bash scripts/tabular_genre2.sh Abalone_reg init_checkpoint=/path/to/model.pt
```

## RLM Scripts

Standalone RLM code metric scripts:

```text
scripts/rlm_base.sh
scripts/rlm_ce.sh
scripts/rlm_ntl_mse.sh
scripts/rlm_ntl_was.sh
scripts/rlm_dist2.sh
scripts/rlm_remax.sh
scripts/rlm_genre2.sh
```

Example:

```bash
GPUS=0,1,2,3 NUM_PROCESSES=4 RLM_DATA_DIR=/path/to/code_metric \
  bash scripts/rlm_genre2.sh apps
```

Small RLM smoke test:

```bash
GPUS=0 NUM_PROCESSES=1 RLM_DATA_DIR=/path/to/code_metric \
MAX_ITEMS=4 EPOCHS=1 BATCH_SIZE=1 USE_WANDB=false \
  bash scripts/rlm_ce.sh apps
```

## GRM Scripts

Standalone GRM scripts:

```text
scripts/grm_sft.sh
scripts/grm_dist2.sh
scripts/grm_remax.sh
scripts/grm_genre2.sh
scripts/grm_eval.sh
```

Example:

```bash
GPUS=0,1,2,3 NUM_PROCESSES=4 \
GRM_MODEL=/path/to/sft_model \
GRM_REF_MODEL=prometheus-eval/prometheus-8x7b-v2.0 \
  bash scripts/grm_genre2.sh
```

GRM RewardBench-style evaluation is exposed through:

```bash
GRM_EVAL_MODEL=/path/to/model GRM_EVAL_NUM_GPUS=1 \
  bash scripts/grm_eval.sh --debug
```

GRM training wrappers pass through extra TRL/script arguments. For a tiny
debugging run, cap the training split explicitly:

```bash
bash scripts/grm_sft.sh --max_train_samples 100
```

The eval wrapper defaults to `--do_not_save`. Local model evaluation requires
`rewardbench`, `fschat`, and `vllm`; API-model evaluation requires the relevant
provider credentials.

## Release Scope

This code release provides run/evaluation code, Exp1 architecture parameters,
and the Exp1 resolved CE hyperparameters for the 100 filtered tasks. It
does not publish model weights, result archives, model cards, or Hub upload
scripts.
The public release notes are:

```text
docs/exp1_model_architectures.md
docs/exp1_task_filtering.md
docs/release_status.md
```

## Shared Options

All wrappers call `scripts/run_main_experiments.sh`. Common variables:

```text
GPUS                  Comma-separated GPU ids. Default: 0
NUM_PROCESSES         Number of accelerate processes. Default: number of GPUS
PORT                  Accelerate main process port. Default: 12325
SEED                  Random seed. Default: 42
OUTPUT_DIR            Output directory. Default: outputs/<mode>/<method>
                      Some legacy Hydra training modules still write method-
                      specific result directories internally.
EPOCHS                Number of training epochs.
BATCH_SIZE            Training batch size.
LR                    Learning rate.
DRY_RUN               Print the command without running it when set to 1.
MAX_ITEMS             Optional sample cap for smoke tests.
USE_WANDB             Disable experiment tracking when set to false.
HF_MODEL              Optional Hugging Face model id or local path for RLM.
HF_LOCAL_DIR          Optional local Hugging Face model directory for RLM.
GRM_REF_MODEL         Teacher/reference model for GRM GenRe2.
                      Default: prometheus-eval/prometheus-8x7b-v2.0.
BETA                  GRM GenRe2 KL/reference weight. Default: 0.1.
```

## Smoke Tests

After installing dependencies, run:

```bash
python scripts/smoke_test_release.py
```

This imports representative tabular, RLM, and GRM modules, runs tiny pointwise
and Riemann tabular jobs on synthetic TALENT-style data, and checks the GRM eval
CLI.

The release verifier can run the same smoke test when requested:

```bash
RUN_SMOKE=1 ./scripts/verify_release.sh
```
