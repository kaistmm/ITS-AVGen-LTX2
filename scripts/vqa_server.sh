#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
REWARD_MODEL="${REWARD_MODEL:-VideoReward}"
ALIGN_MODEL="${ALIGN_MODEL:-JavisScore}"
SERVER_PORT="${SERVER_PORT:-5002}"

echo "Starting reward server..."
echo "  GPU ID: ${GPU_ID}"
echo "  Reward Model: ${REWARD_MODEL}"
echo "  Align Model: ${ALIGN_MODEL}"
echo "  Port: ${SERVER_PORT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python reward_model/vqa_server.py \
  --gpu "${GPU_ID}" \
  --addr "${SERVER_PORT}" \
  --reward_model "${REWARD_MODEL}" \
  --align_model "${ALIGN_MODEL}"
