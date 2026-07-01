#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODULE="${GRM_EVAL_MODULE:-src.generative_reward_models.evaluation.reward_bench_generative}"
MODEL="${GRM_EVAL_MODEL:-${GRM_MODEL:-}}"

if [[ "${1:-}" == "--help" ]]; then
  cmd=(python -m "${MODULE}" --help)
elif [[ -z "${MODEL}" ]]; then
  echo "Set GRM_EVAL_MODEL or GRM_MODEL to a local model path or Hugging Face id." >&2
  exit 2
else
  cmd=(python -m "${MODULE}" --model "${MODEL}" --do_not_save)
  if [[ "${GRM_EVAL_DEBUG:-0}" == "1" ]]; then
    cmd+=(--debug)
  fi
  if [[ -n "${GRM_EVAL_NUM_GPUS:-}" ]]; then
    cmd+=(--num_gpus "${GRM_EVAL_NUM_GPUS}")
  fi
  if [[ -n "${GRM_EVAL_CHAT_TEMPLATE:-}" ]]; then
    cmd+=(--chat_template "${GRM_EVAL_CHAT_TEMPLATE}")
  fi
  if [[ "${GRM_EVAL_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi
  cmd+=("$@")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
