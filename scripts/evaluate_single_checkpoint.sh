#!/bin/bash
# SmolVLM Single Checkpoint Evaluation Script
# Quick evaluation of a single checkpoint

# Default configuration (can be overridden with environment variables)
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/nas/xulei/VLM-R1_Smol/checkpoints/rl/smolvlm-instruct-rec-grpo-lora-optimized}"
DATA_ROOT="${DATA_ROOT:-/nas/xulei/VLM-R1_Smol/rec_jsons_processed}"
IMAGE_ROOT="${IMAGE_ROOT:-/nas/xulei/VLM-R1_Smol/coco}"
LISA_IMAGE_ROOT="${LISA_IMAGE_ROOT:-/nas/xulei/VLM-R1_Smol/lisa}"
OUTPUT_BASE="${OUTPUT_BASE:-/nas/xulei/VLM-R1_Smol/eval_results}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/nas/xulei/VLM-R1_Smol/src/eval/test_rec_smolvlm_updated.py}"

# Default datasets (can be overridden)
DATASETS="${DATASETS:-refcoco_val refcocop_val refcocog_val lisa_grounding_test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_step> [datasets...]"
    echo ""
    echo "Examples:"
    echo "  $0 7400                                    # Evaluate checkpoint-7400 on all datasets"
    echo "  $0 7400 refcoco_val refcocop_val          # Evaluate checkpoint-7400 on specific datasets"
    echo "  BATCH_SIZE=8 $0 7400                      # Use larger batch size"
    echo "  MAX_SAMPLES=100 $0 7400 refcoco_val       # Quick test with 100 samples"
    echo ""
    echo "Available checkpoints:"
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        find $CHECKPOINT_BASE_DIR -name "checkpoint-*" -type d | sort -V | while read checkpoint; do
            echo "  - $(basename $checkpoint)"
        done
    fi
    echo ""
    echo "Available datasets:"
    if [ -d "$DATA_ROOT" ]; then
        find $DATA_ROOT -name "*.json" | while read dataset; do
            basename_file=$(basename $dataset .json)
            echo "  - $basename_file"
        done
    fi
    exit 1
fi

# Get checkpoint step and optional datasets
CHECKPOINT_STEP=$1
shift
if [ $# -gt 0 ]; then
    DATASETS="$@"
fi

# Build checkpoint path
CHECKPOINT_PATH="$CHECKPOINT_BASE_DIR/checkpoint-$CHECKPOINT_STEP"
OUTPUT_DIR="$OUTPUT_BASE/checkpoint-$CHECKPOINT_STEP"

echo "🚀 SmolVLM Single Checkpoint Evaluation"
echo "======================================="
echo "Checkpoint: checkpoint-$CHECKPOINT_STEP"
echo "Path: $CHECKPOINT_PATH"
echo "Datasets: $DATASETS"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
fi
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        find $CHECKPOINT_BASE_DIR -name "checkpoint-*" -type d | sort -V | while read checkpoint; do
            echo "  - $(basename $checkpoint)"
        done
    fi
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python $EVAL_SCRIPT \
    --model_path \"$CHECKPOINT_PATH\" \
    --data_root \"$DATA_ROOT\" \
    --image_root \"$IMAGE_ROOT\" \
    --lisa_image_root \"$LISA_IMAGE_ROOT\" \
    --output_dir \"$OUTPUT_DIR\" \
    --datasets $DATASETS \
    --batch_size $BATCH_SIZE"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo "📊 Starting evaluation..."
echo "Command: $CMD"
echo ""

# Run evaluation
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo ""
    
    # Show quick summary if summary file exists
    SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "📊 Quick Summary:"
        python - << EOF
import json
with open("$SUMMARY_FILE", 'r') as f:
    data = json.load(f)

for dataset, metrics in data['metrics'].items():
    if 'error' not in metrics:
        print(f"  {dataset:20} | Acc@0.5: {metrics['accuracy_at_0.5']:6.2f}% | Acc@0.7: {metrics['accuracy_at_0.7']:6.2f}% | Format: {metrics['format_accuracy']:6.2f}%")
    else:
        print(f"  {dataset:20} | ERROR: {metrics['error']}")
EOF
    fi
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi
