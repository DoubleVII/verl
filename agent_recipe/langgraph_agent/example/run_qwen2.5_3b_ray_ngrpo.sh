#!/bin/bash
set -xeuo pipefail

# MathExpression Training Script
# This script trains a model to solve math expressions with the @ operator using tool calls

# Environment setup
export GPUS_PER_NODE=4
export NNODES=1

# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export RAY_LOGGING_LEVEL=DEBUG
# export HYDRA_FULL_ERROR=1

# # Wandb offline mode - log locally first, sync to wandb.ai later
# export WANDB_MODE=offline
# # Optional: specify local directory for wandb logs (default is ./wandb)
# export WANDB_DIR=/home/yangs/wandb_logs

unset http_proxy
unset https_proxy

# Model path - using local path
model_path=/home/nfs06/yangs/LLM/Qwen/Qwen2.5-3B-Instruct
DATA_ROOT=/home/yangs

# Data paths
DATA_DIR=$DATA_ROOT/data/math_expression_tool
train_files="['${DATA_DIR}/train.parquet']"
test_files="['${DATA_DIR}/test.parquet']"

# Check if dataset exists, if not create it
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Dataset not found, creating..."
    python agent_recipe/langgraph_agent/example/create_dataset.py \
        --train_size 5000 \
        --test_size 500 \
        --output_dir ${DATA_DIR}
fi

# Agent config
agent_loop_config_path=agent_recipe/langgraph_agent/example/agent.yaml

# =================== wandb ===================
# Project settings
project_name='verl-math-expression'
experiment_name="math_expression_tool_qwen2.5_3b"

# Checkpoint directory
default_local_dir=$DATA_ROOT/ckpt/math_expression4/$experiment_name

# ================= algorithm =================
adv_estimator=ngrpo

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=1024
max_response_length=2048
actor_lr=1e-6

train_batch_size=256
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=8
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# =================== logging ===================

# ================= performance =================

infer_tp=2  # vLLM tensor parallel size
train_sp=1  # Ulysses sequence parallel size for actor
offload=true

# Rollout parameters
rollout_engine=vllm
rollout_mode=async
gpu_memory_utilization=0.8

# Sampling params
temperature=1.0
top_p=1.0
top_k=-1

# Training steps
total_epochs=6
test_freq=5
save_freq=500
val_before_train=false

# Performance Related Parameter
use_dynamic_bsz=true
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))


# Submit job using ray
ray job submit \
    --runtime-env=verl/trainer/runtime_env_edited.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=4 \
    $@
