#!/bin/bash
# filepath: run_scripts/run_grpo_rec_smolvlm_lora.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_NAME="smolvlm-instruct-rec-grpo-lora"
REPO_HOME="/nas/xulei/VLM-R1_Smol"

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
export DEBUG_MODE="true"

# Set up wandb
export WANDB_API_KEY="f261eb65ba1dbe4a2ce2ff847c0530fa97db1f8c"
wandb login $WANDB_API_KEY

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${RUN_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --data_seed 42 \
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
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --reward_funcs "accuracy" "format" \
    --num_generations 2 \
    --max_completion_length 2048 \
    --beta 0.04 \
    --temperature 1.0 \
    --num_iterations 1 \
    --epsilon 0.1 \
    --freeze_vision_modules True \
    --val_split_ratio 0.1 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_task_type CAUSAL_LM \
    --lora_target_modules "down_proj" "o_proj" "k_proj" "q_proj" "gate_proj" "up_proj" "v_proj" \
    --dataset-name this_is_not_used \
    --attn_implementation flash_attention_2 \

    2>&1 | tee "${LOG_PATH}"
