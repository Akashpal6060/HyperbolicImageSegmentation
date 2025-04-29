#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run.sh — Launch IDDAW Hyperbolic SegFormer training under nohup
# Usage: ./run.sh [GPU_ID] [BATCH_SIZE] [NUM_EPOCHS]
# -----------------------------------------------------------------------------

# Default settings
GPU_ID=${1:-2}        # default to GPU 2
BATCH_SIZE=${2:-64}   # default batch size 64
NUM_EPOCHS=${3:-500}   # default to 50 epochs

# Export only the desired GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Log file
LOG_FILE="train_IDDAW_gpu${GPU_ID}_bs${BATCH_SIZE}.log"

# Create logs directory if needed
mkdir -p logs
LOG_PATH="logs/$LOG_FILE"

echo "Starting training on GPU $GPU_ID with batch size $BATCH_SIZE for $NUM_EPOCHS epochs"
echo "Logging to $LOG_PATH"

nohup python samples/train.py \
  --mode segmenter \
  --dataset IDDAW \
  --geometry hyperbolic \
  --dim 256 \
  --c 1.0 \
  --batch_size $BATCH_SIZE \
  --slr 1e-3 \
  --num_epochs $NUM_EPOCHS \
  --output_stride 16 \
  --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
  --freeze_bn \
  --gpu 0 \
  --train \
  > "$LOG_PATH" 2>&1 &

echo "Process launched." &> /dev/null