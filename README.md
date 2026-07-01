# GenRe2

Code for the main experiments of GenRe2.

## Environment installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate genre2-main
```

Or install the required packages in an existing environment:

```bash
pip install -r requirements.txt
```

The scripts use `accelerate` for multi-GPU runs. Configure it once before
training:

```bash
accelerate config
```

The release wrappers default to the checked-in DeepSpeed ZeRO-2 config:

```bash
export ACCELERATE_CONFIG=configs/accelerate_deepspeed_zero2.yaml
```

Set `ACCELERATE_CONFIG=` if you want to use your local accelerate default.

Set data and model paths as needed:

```bash
export TABULAR_DATA_DIR=/path/to/talent
export RLM_DATA_DIR=/path/to/code_metric
export GRM_MODEL=/path/to/sft_model
export GRM_REF_MODEL=prometheus-eval/prometheus-8x7b-v2.0
```

Use `DRY_RUN=1` to print the command without starting a run.

## Main experiments

For `accelerate` jobs, `BATCH_SIZE` is passed directly to the training code and
is the per-process DataLoader batch size. The tabular paper runs use
`BATCH_SIZE=128` with 4 accelerate processes.

Common runtime variables:

```bash
export GPUS=0,1,2,3
export NUM_PROCESSES=4
export SEED=42
```

### Tabular regression

Search CE hyperparameters:

```bash
TABULAR_DATA_DIR=/path/to/talent N_TRIALS=25 \
  bash scripts/tabular_search_ce.sh Abalone_reg
```

Run GenRe2:

```bash
GPUS=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=128 TABULAR_DATA_DIR=/path/to/talent \
  bash scripts/tabular_genre2.sh Abalone_reg
```

Other tabular wrappers:

```text
scripts/tabular_pointwise.sh
scripts/tabular_riemann.sh
scripts/tabular_ce.sh
scripts/tabular_ntl_mse.sh
scripts/tabular_ntl_was.sh
scripts/tabular_dist2.sh
scripts/tabular_remax.sh
```

For NTL, DIST2, ReMax, and GenRe2, the default initialization is the CE
checkpoint produced by `scripts/tabular_ce.sh`. To use another checkpoint:

```bash
bash scripts/tabular_genre2.sh Abalone_reg init_checkpoint=/path/to/model.pt
```

### RLM code metric regression

Run GenRe2 on APPS:

```bash
GPUS=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 RLM_DATA_DIR=/path/to/code_metric \
  bash scripts/rlm_genre2.sh apps
```

Small check run:

```bash
GPUS=0 NUM_PROCESSES=1 RLM_DATA_DIR=/path/to/code_metric \
MAX_ITEMS=4 EPOCHS=1 BATCH_SIZE=1 USE_WANDB=false \
  bash scripts/rlm_ce.sh apps
```

Other RLM wrappers:

```text
scripts/rlm_base.sh
scripts/rlm_ce.sh
scripts/rlm_ntl_mse.sh
scripts/rlm_ntl_was.sh
scripts/rlm_dist2.sh
scripts/rlm_remax.sh
```

### Generative reward model

Run GenRe2:

```bash
GPUS=0,1,2,3 NUM_PROCESSES=4 \
GRM_MODEL=/path/to/sft_model \
GRM_REF_MODEL=prometheus-eval/prometheus-8x7b-v2.0 \
  bash scripts/grm_genre2.sh
```

Evaluate:

```bash
GRM_EVAL_MODEL=/path/to/model GRM_EVAL_NUM_GPUS=1 \
  bash scripts/grm_eval.sh --debug
```

Other GRM wrappers:

```text
scripts/grm_sft.sh
scripts/grm_dist2.sh
scripts/grm_remax.sh
```

For a short GRM debug run:

```bash
bash scripts/grm_sft.sh --max_train_samples 100
```

## Citation

```bibtex
@inproceedings{chen2026beyond,
  title={Beyond Token-level Supervision: Unlocking the Potential of Decoding-based Regression via Reinforcement Learning},
  author={Chen, Ming and Tang, Sheng and Tan, Rong-Xi and Li, Ziniu and Chen, Jiacheng and Xue, Ke and Qian, Chao},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  address={Seoul, South Korea},
  year={2026}
}
```
