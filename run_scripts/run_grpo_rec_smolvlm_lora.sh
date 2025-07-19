#!/bin/bash
# filepath: run_scripts/run_grpo_rec_smolvlm_lora.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RUN_NAME="smolvlm-instruct-rec-grpo-lora"
REPO_HOME="/home/peisheng/VLM-R1"

# Using RefCOCO data as you mentioned
data_paths="${REPO_HOME}/rec_jsons_processed/refcoco_train.jsonl"
image_folders="${REPO_HOME}/coco"
model_path="HuggingFaceTB/SmolVLM-Instruct"
is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"

cd ${REPO_HOME}/src/open-r1-multimodal

# Create run directory for logs
mkdir -p ${REPO_HOME}/runs/${RUN_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${RUN_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${RUN_NAME} \
    --model_name_or_path $model_path \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --report_to "wandb" \
    --run_name $RUN_NAME \
    --bf16 \
    --tf32 True \
    --eval_strategy "no" \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --reward_funcs "accuracy" "format" \
    --num_generations 4 \
    --max_completion_length 256 \
    --beta 0.1 \
    --temperature 1.0 \
    --num_iterations 1 \
    --epsilon 0.1 \
    --freeze_vision_modules True \
    --val_split_ratio 0.1 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --dataset-name this_is_not_used
