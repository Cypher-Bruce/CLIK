#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to a JSONL file with context, prompt, and completion fields}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date '+%Y.%m.%d-%H.%M.%S')}"
PG_CONFIG="${PG_CONFIG:-configs/pg_config_qwen3_5_4b.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "${EVAL_DATASET_PATH:-}" ]]; then
  EXTRA_ARGS+=(--eval_dataset_path "$EVAL_DATASET_PATH")
  EXTRA_ARGS+=(--eval_steps "${EVAL_STEPS:-500}")
fi
if [[ -n "${INIT_WEIGHTS_FROM:-}" ]]; then
  EXTRA_ARGS+=(--init_weights_from "$INIT_WEIGHTS_FROM")
fi
if [[ "${USE_DISTILLATION:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--use_distillation)
  EXTRA_ARGS+=(--distill_temperature "${DISTILL_TEMPERATURE:-2.0}")
fi
if [[ "${ENABLE_WANDB:-0}" != "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  train.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name jsonl \
  --dataset_path "$DATASET_PATH" \
  --max_seq_len "${MAX_SEQ_LEN:-8192}" \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --precision bf16 \
  --use_pg \
  --pg_config "$PG_CONFIG" \
  --pg_hidden_strategy "${PG_HIDDEN_STRATEGY:-single}" \
  --gradient_checkpointing \
  --save_steps "${SAVE_STEPS:-500}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --seed "${SEED:-42}" \
  "${EXTRA_ARGS[@]}"
