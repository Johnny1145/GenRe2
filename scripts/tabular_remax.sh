#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-Abalone_reg}"
shift || true
exec bash "${SCRIPT_DIR}/run_main_experiments.sh" tabular remax "${DATASET}" "$@"
