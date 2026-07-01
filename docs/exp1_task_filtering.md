# Exp1 Task Filtering and Search Hyperparameters

This release includes per-task effective hyperparameters for the 100 Exp1
TALENT-style regression tasks. They are resolved from CE Optuna
`best_params.json` files plus the checkpoint-verification logic in the
original `src/RL_reweight_exp/exp1_remax_ce.py` script.

## Task Filter

The 100-task set is reproduced from the later result-table scripts. The
filtering rule is:

1. scan `regression_data_new/*/info.json`;
2. keep datasets with numeric `train_size <= 1000000000`;
3. sort the dataset names lexicographically.

The source implementation is the `check_train_size` and
`collect_metrics_from_directories` path in the original
`find_better_tasks.py` / `analyze_metrics_results_to_big_table.py` scripts.

## Included Hyperparameter Sources

- `results_optuna_ce/`: Effective Exp1 parameters resolved from CE Optuna best_params and the exp1_remax_ce.py results_merge_ce checkpoint verification rule.
  Flat CSV: `docs/exp1_ce_optuna_best_params.csv`
  Consolidated JSON: `docs/exp1_resolved_hyperparameters.json`

The Optuna loader first reads the nested path and then a flat fallback
path:

```text
results_optuna_ce/<dataset>/<dataset>/best_params.json
results_optuna_ce/<dataset>/best_params.json
```

The final released values are not a raw Optuna copy. For each dataset,
the exporter resolves the checkpoint path:

```text
results_merge_ce/<dataset>/<dataset>/checkpoints_42/checkpoint_best/model.pt
```

It then infers `d_model`, `num_decoder_layers`, `dim_feedforward`, and
`hidden_dim` from the checkpoint. If those architecture fields match
Optuna, the final row uses Optuna for `learning_rate`, `base`, `digits`,
and `nhead`, and uses the checkpoint-inferred architecture fields. If
they do not match, the final row uses checkpoint-inferred architecture
fields and the fixed defaults `learning_rate=1e-5`, `base=2`,
`digits=8`, `nhead=4`, and `dropout=0.1`.

`hidden_dim` is stored as a scalar in the files; training code converts
it to `cfg.model.hidden_dims = [hidden_dim]`. The consolidated JSON
stores `hyperparameters[dataset]` as the direct final parameter map.
The CSV and JSON provenance sections also record whether a task used
the `optuna_plus_checkpoint_arch_match` or
`checkpoint_defaults_due_arch_mismatch` branch.
