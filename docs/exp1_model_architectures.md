# Exp1 Model Architecture Parameters

This release open-sources the Exp1 model architecture parameters and the
run/evaluation code. It also includes the resolved CE hyperparameters for the
100 filtered Exp1 tasks. It intentionally does not include model
weights.

## Scope

Exp1 covers TALENT-style tabular regression models. The release wrappers are:

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

## Architecture Summary

| Exp1 model | Architecture family | Release wrapper | Main module |
| --- | --- | --- | --- |
| Pointwise head baseline | MLP scalar regressor | `scripts/tabular_pointwise.sh` | `src.run_tabular_head_baselines --head pointwise` |
| Riemann head baseline | MLP distributional regressor | `scripts/tabular_riemann.sh` | `src.run_tabular_head_baselines --head riemann` |
| CE hyperparameter search | RegressLM MLP encoder-decoder | `scripts/tabular_search_ce.sh` | `src.search.search_ce` |
| CE | RegressLM MLP encoder-decoder | `scripts/tabular_ce.sh` | `src.search.train_search_ce` |
| NTL-MSE | RegressLM MLP encoder-decoder | `scripts/tabular_ntl_mse.sh` | `src.search.train_search_ntl_mse` |
| NTL-WAS | RegressLM MLP encoder-decoder | `scripts/tabular_ntl_was.sh` | `src.search.train_search_ntl_was` |
| DIST2 | RegressLM MLP encoder-decoder | `scripts/tabular_dist2.sh` | `src.search.train_search_dist2` |
| ReMax | RegressLM MLP encoder-decoder | `scripts/tabular_remax.sh` | `src.search.train_search_rl` |
| GenRe2 | RegressLM MLP encoder-decoder | `scripts/tabular_genre2.sh` | `src.search.train_search_rl_expert` |

CE, NTL-MSE, NTL-WAS, DIST2, ReMax, and GenRe2 share the same model
architecture. They differ in objective/training logic, not in the architecture
parameters below.

## Pointwise Head Baseline

| Parameter | Value |
| --- | --- |
| Input features | TALENT numeric/categorical features after preprocessing |
| Hidden layers | `num_layers=1` by release default |
| Hidden width | `hidden_dim=2048` |
| Activation | ReLU |
| Dropout | `0.1` |
| Output | scalar regression value |

Implementation: `PointwiseMLP` in `src/run_tabular_head_baselines.py`.

## Riemann Head Baseline

| Parameter | Value |
| --- | --- |
| Input features | TALENT numeric/categorical features after preprocessing |
| Hidden layers | `num_layers=3` |
| Hidden width | `hidden_dim=2048` |
| Activation | ReLU |
| Dropout | `0.1` |
| Distribution bins | `num_bins=256` |
| Support range | `y_min=-3.0`, `y_max=3.0` |
| Output dimension | `num_bins + 2` |
| Prediction | expected value over learned bin probabilities |

Implementation: `RiemannMLP` in `src/run_tabular_head_baselines.py`.

## RegressLM MLP Encoder-Decoder Models

These parameters apply to CE, NTL-MSE, NTL-WAS, DIST2, ReMax, and GenRe2.

| Parameter | Value |
| --- | --- |
| Encoder type | `mlp` |
| `model.input_dim` | `42` |
| `model.hidden_dims` | `[1024, 1024, 1024]` |
| `model.output_dim` | `256` |
| `model.max_input_len` | `1024` |
| `model.max_num_objs` | `1` |
| `model.d_model` | `256` |
| `model.num_encoder_layers` | `3` |
| `model.num_decoder_layers` | `3` |
| `model.nhead` | `4` |
| `model.dim_feedforward` | `1024` |
| `model.dropout` | `0.1` |

Numeric decoder/tokenization settings:

| Parameter | Value |
| --- | --- |
| Default normalized numeric base | `base=2` |
| Default number of digits | `digits=8` |
| Broad sweep grid used by original scripts | `(2, 8)`, `(4, 4)`, `(6, 4)`, `(8, 3)`, `(10, 3)` |

Per-task resolved CE hyperparameters are included under:

```text
results_optuna_ce/<dataset>/<dataset>/best_params.json
docs/exp1_resolved_hyperparameters.json
docs/exp1_ce_optuna_best_params.csv
docs/exp1_search_hyperparameters_manifest.json
```

The 100-task filtering rule and parameter application rule are documented in
`docs/exp1_task_filtering.md`.

Source Hydra files:

```text
src/conf/experiment/exp5_mlp_encoder_ce.yaml
src/conf/experiment/baseline_ntl.yaml
src/conf/experiment/exp_RL_mlp_encoder_big_batch.yaml
```

## Objective Parameters Kept for Reproduction

The following are not architecture parameters, but they are exposed by wrappers
because they affect reproduction:

| Model | Parameter | Default |
| --- | --- | --- |
| ReMax | `reinforce.num_samples` | `16` |
| ReMax | `reinforce.temperature` | `1.0` |
| ReMax | `reinforce.reward_scale` | `1.0` |
| ReMax | `reinforce.one_sided_norm` | `true` |
| GenRe2 | `OPD_WEIGHT` / `reinforce.expert_ce_weight` | `0.05` |

`OPD_WEIGHT=0.05` matches the shared `exp_RL_mlp_encoder_big_batch` Hydra
configuration and is the default used by the cleaned release wrapper. Override
it only when intentionally running ablations.

## Source Evidence

Architecture values were checked against the original pre-cleanup configs and
scripts listed below. The release equivalents are under `src/conf` and
`scripts/`.

```text
src/conf/experiment/exp5_mlp_encoder_ce.yaml
src/conf/experiment/baseline_ntl.yaml
src/conf/experiment/exp_RL_mlp_encoder_big_batch.yaml
scripts/tabular_ce.sh
scripts/tabular_ntl_mse.sh
scripts/tabular_ntl_was.sh
scripts/tabular_dist2.sh
scripts/tabular_remax.sh
```
