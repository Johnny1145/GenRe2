# GenRe2 Code and Exp1 Hyperparameter Release Status

Status date: 2026-06-30.

This release is scoped to code, Exp1 model architecture parameter disclosure,
and Exp1 resolved CE hyperparameter disclosure for the 100 filtered
tasks.
It intentionally does not include model weights, model cards, Hub upload scripts, or weight
manifests.

## Included

- Main experiment training/evaluation code for tabular regression, RLM code
  metric regression, and GRM training/evaluation.
- Shell wrappers under `scripts/` for reproducible command construction.
- `scripts/grm_eval.sh` for GRM RewardBench-style evaluation.
- `scripts/smoke_test_release.py` for lightweight runtime validation.
- Exp1 tabular model architecture parameters in `docs/exp1_model_architectures.md`.
- Exp1 resolved CE hyperparameters for 100 tasks:
  `results_optuna_ce/<dataset>/<dataset>/best_params.json`,
  `docs/exp1_resolved_hyperparameters.json`,
  `docs/exp1_ce_optuna_best_params.csv`, and
  `docs/exp1_search_hyperparameters_manifest.json`.
- Exp1 task filtering and parameter application rules in
  `docs/exp1_task_filtering.md`.
- Environment files: `requirements.txt` and `environment.yml`.
- Apache-2.0 code license in `LICENSE`.

## Not Included

- GRM or RLM model weights.
- Weight upload scripts.
- Hugging Face model cards.
- Result archives or result dataset cards.
- Cached checkpoints, generated figures, temporary tests, and local result
  dumps.

## Verification

Run:

```bash
./scripts/verify_release.sh
```

The verifier checks:

- shell syntax for all release wrappers;
- README and Exp1 architecture coverage for every tabular Exp1 model;
- exactly 100 resolved CE parameter files and matching JSON/manifest/CSV rows;
- dry-run command generation for representative tabular, RLM, GRM training, and
  GRM evaluation runs;
- absence of weight publication artifacts in the release tree;
- absence of large files above 10 MB;
- absence of known hard-coded secret patterns.
- optional runtime smoke tests when `RUN_SMOKE=1`.

Run the runtime smoke test in an environment with the dependencies installed:

```bash
python scripts/smoke_test_release.py
```

## Current Archive

After editing, refresh the archive from the parent directory of this release:

```bash
cd ..
tar -czf genre2_main_experiments_release_20260630.tar.gz genre2_main_experiments
sha256sum genre2_main_experiments_release_20260630.tar.gz
```
