#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODE="${1:-}"
METHOD="${2:-}"

if [[ -z "${MODE}" || -z "${METHOD}" ]]; then
  cat <<'USAGE'
Usage:
  bash scripts/run_main_experiments.sh tabular <pointwise|riemann|search_ce|ce|ntl_mse|ntl_was|dist2|remax|genre2> [dataset]
  bash scripts/run_main_experiments.sh rlm <base|ce|ntl_mse|ntl_was|dist2|remax|genre2> [dataset] [extra hydra args...]
  bash scripts/run_main_experiments.sh grm <sft|dist2|remax|genre2>

Common environment variables:
  GPUS                  Comma-separated GPU ids. Default: 0
  NUM_PROCESSES         Number of accelerate processes. Default: number of GPUS
  ACCELERATE_CONFIG     Accelerate config. Default: configs/accelerate_deepspeed_zero2.yaml
                        Set ACCELERATE_CONFIG= to use your local accelerate default.
  SEED                  Random seed. Default: 42
  OUTPUT_DIR            Output directory. Default: outputs/<mode>/<method>
  BATCH_SIZE            Training batch size passed to the training code.
                        For accelerate-launched jobs this is per process.
  MAX_ITEMS             Optional sample cap for smoke tests.

Tabular environment variables:
  TABULAR_DATA_DIR      TALENT-style data directory. Default: data/talent
  BASE                  Normalized tokenizer base. Default: 2
  DIGITS                Number of normalized digits. Default: 8

RLM environment variables:
  RLM_DATA_DIR          Code metric jsonl directory. Default: data/code_metric
  HF_MODEL              Hugging Face model id or path. Default: config value.
  HF_LOCAL_DIR          Local Hugging Face model directory. Overrides HF_MODEL when set.

GRM environment variables:
  GRM_MODEL             Model path or HF id.
  GRM_REF_MODEL         Teacher/reference model path or HF id for GenRe2.
USAGE
  exit 1
fi

shift 2

GPUS="${GPUS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F',' '{print NF}' <<<"${GPUS}")}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG-configs/accelerate_deepspeed_zero2.yaml}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODE}/${METHOD}}"
PORT="${PORT:-12325}"

run_accelerate() {
  local launch_args=(launch)
  if [[ -n "${ACCELERATE_CONFIG}" ]]; then
    if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
      echo "ACCELERATE_CONFIG does not exist: ${ACCELERATE_CONFIG}" >&2
      exit 2
    fi
    launch_args+=("--config_file=${ACCELERATE_CONFIG}")
  fi
  launch_args+=(
    "--num_processes=${NUM_PROCESSES}"
    "--main_process_port=${PORT}"
  )
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${GPUS}" accelerate
    "${launch_args[@]}"
    "$@"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${GPUS}" accelerate "${launch_args[@]}" "$@"
  fi
}

run_python() {
  local cmd=(python "$@")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    python "$@"
  fi
}

run_tabular() {
  local dataset="${1:-Abalone_reg}"
  if [[ "$#" -gt 0 ]]; then
    shift
  fi
  local extra_args=("$@")
  local data_dir="${TABULAR_DATA_DIR:-data/talent}"
  local base="${BASE:-2}"
  local digits="${DIGITS:-8}"

  case "${METHOD}" in
    pointwise)
      run_python -m src.run_tabular_head_baselines \
        --head pointwise \
        --data-dir "${data_dir}" \
        --dataset "${dataset}" \
        --results-dir "${OUTPUT_DIR}" \
        --epochs "${EPOCHS:-100}" \
        --batch-size "${BATCH_SIZE:-128}" \
        --hidden-dim "${HIDDEN_DIM:-2048}" \
        --learning-rate "${LR:-1e-5}"
      ;;
    riemann)
      run_python -m src.run_tabular_head_baselines \
        --head riemann \
        --data-dir "${data_dir}" \
        --dataset "${dataset}" \
        --results-dir "${OUTPUT_DIR}" \
        --epochs "${EPOCHS:-100}" \
        --batch-size "${BATCH_SIZE:-128}" \
        --hidden-dim "${HIDDEN_DIM:-2048}" \
        --num-layers "${NUM_LAYERS:-3}" \
        --num-bins "${NUM_BINS:-256}" \
        --learning-rate "${LR:-1e-5}"
      ;;
    search_ce)
      run_accelerate -m src.search.search_ce \
        +experiment=exp5_mlp_encoder_ce \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        use_optuna=true \
        optuna.n_trials="${N_TRIALS:-25}" \
        num_epochs="${EPOCHS:-200}" \
        batch_size="${BATCH_SIZE:-128}" \
        seed="${SEED}" \
        "${extra_args[@]}"
      ;;
    ce)
      run_accelerate -m src.search.train_search_ce \
        +experiment=exp5_mlp_encoder_ce \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        num_epochs="${EPOCHS:-200}" \
        batch_size="${BATCH_SIZE:-128}" \
        learning_rate="${LR:-5e-5}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    ntl_mse)
      run_accelerate -m src.search.train_search_ntl_mse \
        +experiment=baseline_ntl \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        num_epochs="${EPOCHS:-100}" \
        batch_size="${BATCH_SIZE:-128}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    ntl_was)
      run_accelerate -m src.search.train_search_ntl_was \
        +experiment=baseline_ntl \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        num_epochs="${EPOCHS:-100}" \
        batch_size="${BATCH_SIZE:-128}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    dist2)
      run_accelerate -m src.search.train_search_dist2 \
        +experiment=baseline_ntl \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        num_epochs="${EPOCHS:-100}" \
        batch_size="${BATCH_SIZE:-128}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    remax)
      run_accelerate -m src.search.train_search_rl \
        +experiment=exp_RL_mlp_encoder_big_batch \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        num_epochs="${EPOCHS:-100}" \
        batch_size="${BATCH_SIZE:-128}" \
        learning_rate="${LR:-1e-5}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    genre2)
      run_accelerate -m src.search.train_search_rl_expert \
        +experiment=exp_RL_mlp_encoder_big_batch \
        dataset.name="'${dataset}'" \
        dataset.params.data_dir="${data_dir}" \
        reinforce.expert_ce_weight="${OPD_WEIGHT:-0.05}" \
        num_epochs="${EPOCHS:-100}" \
        batch_size="${BATCH_SIZE:-128}" \
        learning_rate="${LR:-1e-5}" \
        seed="${SEED}" \
        base="${base}" \
        digits="${digits}" \
        "${extra_args[@]}"
      ;;
    *)
      echo "Unknown tabular method: ${METHOD}" >&2
      exit 2
      ;;
  esac
}

run_rlm() {
  local dataset="${1:-apps}"
  if [[ "$#" -gt 0 ]]; then
    shift
  fi
  local extra_args=("$@")
  local data_dir="${RLM_DATA_DIR:-data/code_metric}"
  local module=""
  local batch_size="${BATCH_SIZE:-16}"
  local lr="${LR:-5e-6}"

  case "${METHOD}" in
    base) module="src.rlm_exp.zero_shot" ;;
    ce) module="src.rlm_exp.train_ce" ;;
    ntl_mse) module="src.rlm_exp.train_ntl_new" ;;
    ntl_was) module="src.rlm_exp.train_ntl_was" ;;
    dist2) module="src.rlm_exp.train_DIST2" ;;
    remax)
      module="src.rlm_exp.train_rl_new"
      batch_size="${BATCH_SIZE:-8}"
      lr="${LR:-1e-6}"
      ;;
    genre2)
      module="src.rlm_exp.train_rl_expert"
      batch_size="${BATCH_SIZE:-8}"
      lr="${LR:-1e-6}"
      ;;
    *)
      echo "Unknown RLM method: ${METHOD}" >&2
      exit 2
      ;;
  esac

  local args=(
    -m "${module}"
    dataset.task="'${dataset}'" \
    dataset.data_dir="${data_dir}" \
    num_epochs="${EPOCHS:-20}" \
    batch_size="${batch_size}" \
    learning_rate="${lr}" \
    save_every_n_epochs="${SAVE_EVERY_N_EPOCHS:-20}" \
    seed="${SEED}" \
    use_wandb="${USE_WANDB:-false}"
  )

  if [[ -n "${MAX_ITEMS:-}" ]]; then
    args+=(+dataset.max_items="${MAX_ITEMS}")
  fi
  if [[ -n "${HF_MODEL:-}" ]]; then
    args+=(hf.model_name_or_path="${HF_MODEL}")
  fi
  if [[ -n "${HF_LOCAL_DIR:-}" ]]; then
    args+=(+hf.local_dir="${HF_LOCAL_DIR}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    args+=("${extra_args[@]}")
  fi

  run_accelerate "${args[@]}"
}

run_grm() {
  local extra_args=("$@")
  local model="${GRM_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
  local output="${OUTPUT_DIR}"
  local module=""
  local per_device_batch="${BATCH_SIZE:-16}"
  local grad_accum="${GRAD_ACCUM:-1}"
  local epochs="${EPOCHS:-2}"

  case "${METHOD}" in
    sft)
      module="src.generative_reward_models.trainer.sft_Mistral"
      ;;
    dist2)
      module="src.generative_reward_models.trainer.Dist2_Mistral"
      ;;
    remax)
      module="src.generative_reward_models.trainer.ReMax_Mistral"
      per_device_batch="${BATCH_SIZE:-16}"
      ;;
    genre2)
      module="src.generative_reward_models.trainer.ReMax_expert_Mistral"
      per_device_batch="${BATCH_SIZE:-4}"
      grad_accum="${GRAD_ACCUM:-4}"
      epochs="${EPOCHS:-1}"
      ;;
    *)
      echo "Unknown GRM method: ${METHOD}" >&2
      exit 2
      ;;
  esac

  local args=(
    -m "${module}"
    --model_name_or_path="${model}"
    --output_dir="${output}"
    --learning_rate="${LR:-1e-5}"
    --num_train_epochs="${epochs}"
    --optim="${OPTIMIZER:-paged_adamw_32bit}"
    --per_device_train_batch_size="${per_device_batch}"
    --gradient_accumulation_steps="${grad_accum}"
    --report_to="none"
    --logging_steps="${LOGGING_STEPS:-1}"
    --save_strategy="epoch"
    --max_seq_length="${MAX_SEQ_LENGTH:-2048}"
  )

  if [[ "${METHOD}" == "remax" || "${METHOD}" == "genre2" ]]; then
    args+=(--num_generations "${NUM_GENERATIONS:-4}")
    if [[ "${METHOD}" == "genre2" ]]; then
      args+=(--generation_batch_size "${GENERATION_BATCH_SIZE:-32}")
    else
      args+=(--generation_batch_size "${GENERATION_BATCH_SIZE:-64}")
    fi
  fi

  if [[ "${METHOD}" == "genre2" ]]; then
    args+=(--beta "${BETA:-0.1}")
    args+=(--ref_model_name_or_path "${GRM_REF_MODEL:-prometheus-eval/prometheus-8x7b-v2.0}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    args+=("${extra_args[@]}")
  fi

  run_accelerate "${args[@]}"
}

case "${MODE}" in
  tabular) run_tabular "$@" ;;
  rlm) run_rlm "$@" ;;
  grm) run_grm "$@" ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
