# GenRe2

Code for the main GenRe2 experiments.

The repository covers:

- tabular regression on TALENT-style datasets
- RLM code metric regression
- generative reward model training and evaluation
- Exp1 tabular architecture notes
- resolved CE hyperparameters for the 100 filtered Exp1 tabular tasks

Datasets, trained checkpoints, result dumps, plots, and appendix-only ablations
are not included here. No model weights are included.

## Layout

```text
src/
  run_tabular_head_baselines.py   Pointwise and Riemann tabular baselines
  search/                         Tabular CE, NTL, DIST2, ReMax, and GenRe2
  rlm_exp/                        RLM code metric experiments
  generative_reward_models/       GRM training and RewardBench-style eval
  model/                          RegressLM model code
  data/                           Dataset loaders
  utils/                          Losses, RL objectives, checkpoint helpers
scripts/
  run_main_experiments.sh         Shared wrapper implementation
  smoke_test_release.py           Small import/runtime smoke tests
  verify_release.sh               Release sanity checks
docs/
  exp1_model_architectures.md
  exp1_task_filtering.md
  exp1_resolved_hyperparameters.json
  exp1_ce_optuna_best_params.csv
  exp1_search_hyperparameters_manifest.json
results_optuna_ce/
  <dataset>/<dataset>/best_params.json
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate genre2-main
```

Or install into an existing environment:

```bash
pip install -r requirements.txt
```

The training wrappers use `accelerate`. Configure it once if your environment
does not already have a default config:

```bash
accelerate config
```

Set `DRY_RUN=1` to print commands without launching training.

## Data and models

Default relative data locations:

```text
data/talent
data/code_metric
```

Override them with environment variables:

```bash
export TABULAR_DATA_DIR=/path/to/talent
export RLM_DATA_DIR=/path/to/code_metric
```

GRM runs also need a model path:

```bash
export GRM_MODEL=/path/to/sft_or_base_model
export GRM_REF_MODEL=prometheus-eval/prometheus-8x7b-v2.0
```

Check the licenses and terms for any third-party datasets or base models before
redistributing derived files.

## Tabular experiments

The Exp1 architecture settings are in:

```text
docs/exp1_model_architectures.md
```

CE hyperparameters are searched with Optuna:

```bash
TABULAR_DATA_DIR=/path/to/talent N_TRIALS=25 \
  bash scripts/tabular_search_ce.sh Abalone_reg
```

The checked-in Exp1 CE hyperparameter files are:

```text
results_optuna_ce/<dataset>/<dataset>/best_params.json
docs/exp1_resolved_hyperparameters.json
docs/exp1_ce_optuna_best_params.csv
docs/exp1_search_hyperparameters_manifest.json
docs/exp1_task_filtering.md
```

The 100-task list is built from `regression_data_new/*/info.json`, keeping
datasets with numeric `train_size <= 1000000000`, then sorting dataset names
lexicographically. The resolution rule follows the original
`src/RL_reweight_exp/exp1_remax_ce.py` logic: read Optuna `best_params.json`,
infer architecture fields from the `results_merge_ce` checkpoint when possible,
and fall back to fixed defaults for mismatched checkpoint architectures.

Tabular wrappers:

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

NTL, DIST2, ReMax, and GenRe2 initialize from the CE checkpoint produced by
`scripts/tabular_ce.sh`:

```text
results_search_mlp_encoder_ce/<dataset>/<dataset>/checkpoints_<seed>/checkpoint_best/model.pt
```

Use a different checkpoint with a Hydra override:

```bash
bash scripts/tabular_genre2.sh Abalone_reg init_checkpoint=/path/to/model.pt
```

## RLM experiments

Wrappers:

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

Small check run:

```bash
GPUS=0 NUM_PROCESSES=1 RLM_DATA_DIR=/path/to/code_metric \
MAX_ITEMS=4 EPOCHS=1 BATCH_SIZE=1 USE_WANDB=false \
  bash scripts/rlm_ce.sh apps
```

## GRM experiments

Wrappers:

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

Evaluation:

```bash
GRM_EVAL_MODEL=/path/to/model GRM_EVAL_NUM_GPUS=1 \
  bash scripts/grm_eval.sh --debug
```

Extra TRL/script arguments can be passed through the wrappers. For example:

```bash
bash scripts/grm_sft.sh --max_train_samples 100
```

Local GRM evaluation needs `rewardbench`, `fschat`, and `vllm`. API-model
evaluation needs the relevant provider credentials.

## Common wrapper options

All wrappers call `scripts/run_main_experiments.sh`.

```text
GPUS                  Comma-separated GPU ids. Default: 0
NUM_PROCESSES         Number of accelerate processes. Default: number of GPUS
PORT                  Accelerate main process port. Default: 12325
SEED                  Random seed. Default: 42
OUTPUT_DIR            Output directory. Default: outputs/<mode>/<method>
EPOCHS                Number of training epochs
BATCH_SIZE            Training batch size
LR                    Learning rate
DRY_RUN               Print the command without running it when set to 1
MAX_ITEMS             Optional sample cap for smoke tests
USE_WANDB             Disable experiment tracking when set to false
HF_MODEL              Optional Hugging Face model id or local path for RLM
HF_LOCAL_DIR          Optional local Hugging Face model directory for RLM
GRM_REF_MODEL         Teacher/reference model for GRM GenRe2
BETA                  GRM GenRe2 KL/reference weight. Default: 0.1
```

Some legacy Hydra modules still write to method-specific result directories
internally.

## Checks

Run the release checks:

```bash
./scripts/verify_release.sh
```

After installing dependencies, run the smoke test:

```bash
python scripts/smoke_test_release.py
```

The full verifier can include the smoke test:

```bash
RUN_SMOKE=1 ./scripts/verify_release.sh
```

The smoke test imports representative tabular, RLM, and GRM modules, runs tiny
pointwise and Riemann jobs on synthetic TALENT-style data, and checks the GRM
eval CLI.

## License

Code is under the Apache License 2.0. See `LICENSE`.

The partial RLM model card uses MIT metadata to match the upstream public base
checkpoint `akhauriyash/RLM-GemmaS-Code-v0`.
