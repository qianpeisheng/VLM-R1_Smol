#!/bin/bash
# Quick test script for SmolVLM - minimal training for validation

export CUDA_VISIBLE_DEVICES=1

RUN_NAME="smolvlm-quick-test"
REPO_HOME="/nas/xulei/VLM-R1_Smol"

# Using RefCOCO data as you mentioned
data_paths="${REPO_HOME}/rec_jsons_processed/refcoco_train.jsonl"
image_folders="${REPO_HOME}/coco"
model_path="HuggingFaceTB/SmolVLM-Instruct"
is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"

cd ${REPO_HOME}/src/open-r1-multimodal

echo "🚀 Starting SmolVLM quick test..."
echo "This will run for just 10 steps to verify everything works"

# Create run directory for logs
mkdir -p ${REPO_HOME}/runs/${RUN_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${RUN_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# Minimal training for testing
python src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/output/$RUN_NAME \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --report_to "none" \
    --bf16 \
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
    --max_steps 10

echo "✅ Quick test completed! Check the output above for any errors."
