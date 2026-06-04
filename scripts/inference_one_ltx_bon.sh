#!/usr/bin/env bash
set -euo pipefail

# Default parameters
GPU_ID="${1:-0}"
NSHARD="${2:-1}"
SHARD="${3:-0}"
CONFIG_FILE="${CONFIG_FILE:-$(dirname "$0")/inference_one_ltx_bon_config.json}"

# Validate config file
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# Load BON configuration from JSON
eval "$(
  python - <<'PY' "$CONFIG_FILE"
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)

for key, value in config.items():
    print(f': "${{{key}:={shlex.quote(str(value))}}}"')
PY
)"

# Override defaults with environment variables
OUTPUT_DIR="${OUTPUT_DIR:-./results/bon_output}"
BON_SAMPLES="${BON_SAMPLES:-5}"
BON_SERVER_PORT="${BON_SERVER_PORT:-5002}"

# Log configuration
echo "BON Configuration:"
echo "  Aggregation Method: ${BON_AGGREGATION_METHOD}"
echo "  Samples: ${BON_SAMPLES}"
echo "  Reward Key: ${BON_REWARD_KEY}"
echo "  Align Key: ${BON_ALIGN_KEY}"
echo "  Config File: ${CONFIG_FILE}"

# Run inference with BON
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/LTX2.3/ltx-2.3-22b-distilled.safetensors}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
BON_SAMPLES="${BON_SAMPLES}" \
BON_SERVER_PORT="${BON_SERVER_PORT}" \
BON_REWARD_KEY="${BON_REWARD_KEY}" \
BON_ALIGN_KEY="${BON_ALIGN_KEY}" \
BON_REWARD_WEIGHT="${BON_REWARD_WEIGHT}" \
BON_ALIGN_WEIGHT="${BON_ALIGN_WEIGHT}" \
BON_KEEP_CANDIDATES="${BON_KEEP_CANDIDATES}" \
BON_AGGREGATION_METHOD="${BON_AGGREGATION_METHOD}" \
BON_ARW_LR="${BON_ARW_LR}" \
BON_ARW_MAX_ITER="${BON_ARW_MAX_ITER}" \
BON_ARW_OPTIMIZER="${BON_ARW_OPTIMIZER}" \
bash "$(dirname "$0")/inference_one_ltx.sh" "${GPU_ID}" "${NSHARD}" "${SHARD}"
