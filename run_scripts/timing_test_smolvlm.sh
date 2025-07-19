#!/bin/bash
# Timing test script for SmolVLM to estimate full training time

# Check if GPU count is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_gpus> [batch_size_per_gpu] [steps_for_timing]"
    echo "Example: $0 1 2 50  # 1 GPU, batch size 2, 50 steps for timing"
    exit 1
fi

NUM_GPUS=$1
BATCH_SIZE=${2:-2}  # Default batch size per GPU
TIMING_STEPS=${3:-50}  # Default steps for timing

echo "🚀 Starting SmolVLM timing test..."
echo "Configuration: ${NUM_GPUS} GPU(s), batch_size=${BATCH_SIZE}, timing_steps=${TIMING_STEPS}"

# Set GPU visibility based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
    TORCHRUN_CMD=""
elif [ $NUM_GPUS -le 8 ]; then
    # Create comma-separated list: 0,1,2,...,N-1
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    TORCHRUN_CMD="torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345"
else
    echo "Error: This script supports up to 8 GPUs"
    exit 1
fi

RUN_NAME="smolvlm-timing-${NUM_GPUS}gpu-bs${BATCH_SIZE}"
REPO_HOME="/home/peisheng/VLM-R1"

# Using RefCOCO data
data_paths="${REPO_HOME}/rec_jsons_processed/refcoco_train.jsonl"
image_folders="${REPO_HOME}/coco"
model_path="HuggingFaceTB/SmolVLM-Instruct"
is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"

cd ${REPO_HOME}/src/open-r1-multimodal

# Create run directory for logs
mkdir -p ${REPO_HOME}/runs/${RUN_NAME}/log

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date)"

# Calculate total samples and estimate based on timing run
TOTAL_SAMPLES=120624
SAMPLES_PER_STEP=$((NUM_GPUS * BATCH_SIZE))
TOTAL_STEPS_NEEDED=$((TOTAL_SAMPLES / SAMPLES_PER_STEP))

echo "Dataset info:"
echo "  Total training samples: ${TOTAL_SAMPLES}"
echo "  Samples per step: ${SAMPLES_PER_STEP} (${NUM_GPUS} GPUs × ${BATCH_SIZE} batch size)"
echo "  Total steps needed for full epoch: ${TOTAL_STEPS_NEEDED}"
echo "  Timing with ${TIMING_STEPS} steps..."

# Run the timing test
if [ $NUM_GPUS -eq 1 ]; then
    python src/open_r1/grpo_jsonl.py \
        --use_vllm False \
        --output_dir ${REPO_HOME}/runs/${RUN_NAME} \
        --model_name_or_path $model_path \
        --data_file_paths $data_paths \
        --image_folders $image_folders \
        --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
        --task_type $TASK_TYPE \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing false \
        --logging_steps 10 \
        --save_strategy "no" \
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --report_to "none" \
        --bf16 \
        --attn_implementation flash_attention_2 \
        --tf32 True \
        --eval_strategy "no" \
        --dataloader_num_workers 0 \
        --remove_unused_columns False \
        --reward_funcs "accuracy" "format" \
        --num_generations 2 \
        --max_completion_length 128 \
        --beta 0.01 \
        --temperature 1.0 \
        --num_iterations 1 \
        --epsilon 0.1 \
        --freeze_vision_modules True \
        --val_split_ratio 0.1 \
        --use_peft true \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules "q_proj" "v_proj" \
        --dataset-name this_is_not_used \
        --dataloader_pin_memory False \
        --max_steps $TIMING_STEPS
else
    $TORCHRUN_CMD src/open_r1/grpo_jsonl.py \
        --use_vllm False \
        --output_dir ${REPO_HOME}/runs/${RUN_NAME} \
        --model_name_or_path $model_path \
        --data_file_paths $data_paths \
        --image_folders $image_folders \
        --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
        --task_type $TASK_TYPE \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing true \
        --logging_steps 10 \
        --save_strategy "no" \
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --report_to "none" \
        --bf16 \
        --attn_implementation flash_attention_2 \
        --tf32 True \
        --eval_strategy "no" \
        --dataloader_num_workers 0 \
        --remove_unused_columns False \
        --reward_funcs "accuracy" "format" \
        --num_generations 2 \
        --max_completion_length 128 \
        --beta 0.01 \
        --temperature 1.0 \
        --num_iterations 1 \
        --epsilon 0.1 \
        --freeze_vision_modules True \
        --val_split_ratio 0.1 \
        --use_peft true \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules "q_proj" "v_proj" \
        --dataset-name this_is_not_used \
        --dataloader_pin_memory False \
        --max_steps $TIMING_STEPS
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============ TIMING RESULTS ============"
echo "End time: $(date)"
echo "Timing test duration: ${DURATION} seconds ($(echo "scale=2; $DURATION/60" | bc) minutes)"
echo "Time per step: $(echo "scale=2; $DURATION/$TIMING_STEPS" | bc) seconds"
echo ""

# Calculate estimates for full training
TIME_PER_STEP=$(echo "scale=2; $DURATION/$TIMING_STEPS" | bc)
FULL_EPOCH_TIME=$(echo "scale=0; $TIME_PER_STEP * $TOTAL_STEPS_NEEDED" | bc)
FULL_EPOCH_HOURS=$(echo "scale=2; $FULL_EPOCH_TIME / 3600" | bc)

echo "============ FULL TRAINING ESTIMATES ============"
echo "Configuration: ${NUM_GPUS} GPU(s), batch_size=${BATCH_SIZE} per GPU"
echo "Total steps needed: ${TOTAL_STEPS_NEEDED}"
echo "Estimated time per step: ${TIME_PER_STEP} seconds"
echo "Estimated full epoch time: ${FULL_EPOCH_TIME} seconds"
echo "Estimated full epoch time: ${FULL_EPOCH_HOURS} hours"
echo ""

# For typical RL training (3-5 epochs)
EPOCHS_3=$(echo "scale=2; $FULL_EPOCH_HOURS * 3" | bc)
EPOCHS_5=$(echo "scale=2; $FULL_EPOCH_HOURS * 5" | bc)
echo "Estimated time for 3 epochs: ${EPOCHS_3} hours"
echo "Estimated time for 5 epochs: ${EPOCHS_5} hours"
echo "=========================================="
