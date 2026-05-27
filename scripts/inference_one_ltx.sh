#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/LTX2.3/ltx-2.3-22b-dev.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-./checkpoints/LTX2/gemma3}"
PROMPT_FILE="${PROMPT_FILE:-./data/JavisBench/JavisBench-mini-ltx2.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/test_output}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="${PYTHONPATH:-$(pwd)/packages/ltx-pipelines/src:$(pwd)/packages/ltx-core/src}"
USE_FP8_CAST="${USE_FP8_CAST:-1}"
BATCH_MODEL_MODE="${BATCH_MODEL_MODE:-reuse}"
SEED="${SEED:-42}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-}"
PROMPT_EMBEDS_PATH="${PROMPT_EMBEDS_PATH:-}"
NEGATIVE_PROMPT_EMBEDS_PATH="${NEGATIVE_PROMPT_EMBEDS_PATH:-}"
BON_SAMPLES="${BON_SAMPLES:-1}"
BON_SERVER_PORT="${BON_SERVER_PORT:-5002}"
BON_REWARD_KEY="${BON_REWARD_KEY:-scores}"
BON_ALIGN_KEY="${BON_ALIGN_KEY:-None}"
BON_REWARD_WEIGHT="${BON_REWARD_WEIGHT:-1.0}"
BON_ALIGN_WEIGHT="${BON_ALIGN_WEIGHT:-0.0}"
BON_TOPK_MIN="${BON_TOPK_MIN:-0.4}"
BON_KEEP_CANDIDATES="${BON_KEEP_CANDIDATES:-0}"
BON_AGGREGATION_METHOD="${BON_AGGREGATION_METHOD:-weighted}"
BON_ARW_LR="${BON_ARW_LR:-0.05}"
BON_ARW_MAX_ITER="${BON_ARW_MAX_ITER:-1}"
BON_ARW_OPTIMIZER="${BON_ARW_OPTIMIZER:-Adam}"

CMD=(
  "$PYTHON_BIN" -m ltx_pipelines.ti2vid_one_stage
  --checkpoint-path "$CHECKPOINT_PATH"
  --gemma-root "$GEMMA_ROOT"
  --prompt-file "$PROMPT_FILE"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --skip-existing
  --batch-model-mode "$BATCH_MODEL_MODE"
  --num-shards "$NSHARD"
  --shard-index "$SHARD"
)

if [[ -n "$PROMPT_EMBEDS_DIR" ]]; then
  CMD+=(--prompt-embeds-dir "$PROMPT_EMBEDS_DIR")
fi

if [[ -n "$PROMPT_EMBEDS_PATH" ]]; then
  CMD+=(--prompt-embeds-path "$PROMPT_EMBEDS_PATH")
fi

if [[ -n "$NEGATIVE_PROMPT_EMBEDS_PATH" ]]; then
  CMD+=(--negative-prompt-embeds-path "$NEGATIVE_PROMPT_EMBEDS_PATH")
fi

if [[ "$USE_FP8_CAST" == "1" ]]; then
  CMD+=(--quantization fp8-cast)
fi

if [[ "$BON_SAMPLES" -gt 1 ]]; then
  echo "BON runtime: aggregation=${BON_AGGREGATION_METHOD} samples=${BON_SAMPLES} reward=${BON_REWARD_KEY} align=${BON_ALIGN_KEY}"
  CMD+=(
    --bon-samples "$BON_SAMPLES"
    --bon-server-port "$BON_SERVER_PORT"
    --bon-reward-key "$BON_REWARD_KEY"
    --bon-reward-weight "$BON_REWARD_WEIGHT"
    --bon-topk-min "$BON_TOPK_MIN"
    --bon-aggregation-method "$BON_AGGREGATION_METHOD"
    --bon-arw-lr "$BON_ARW_LR"
    --bon-arw-max-iter "$BON_ARW_MAX_ITER"
    --bon-arw-optimizer "$BON_ARW_OPTIMIZER"
  )
  if [[ -n "$BON_ALIGN_KEY" && "$BON_ALIGN_KEY" != "None" ]]; then
    CMD+=(--bon-align-key "$BON_ALIGN_KEY" --bon-align-weight "$BON_ALIGN_WEIGHT")
  fi
  if [[ "$BON_KEEP_CANDIDATES" == "1" ]]; then
    CMD+=(--bon-keep-candidates)
  fi
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
PYTHONPATH="$PYTHONPATH" \
"${CMD[@]}"
