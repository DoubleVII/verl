#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Data and checkpoints follow run_train.mt.v19.sh and can be overridden per run.
TRAIN_FILES=${TRAIN_FILES:-/home/nfs06/yangs/data/parquet_data/training_data/towerx_v2_all_mt-code-block-think.verl/train.parquet}
VAL_FILES=${VAL_FILES:-/home/nfs06/yangs/data/parquet_data/training_data/towerx_v2_all_mt-code-block-think.verl/test_zh.parquet}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/home/nfs06/yangs/ckpt/Qwen/Qwen3.5-4B}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/home/nfs06/yangs/ckpt/Qwen/Qwen2.5-3B/verl/genrm/v13_step150}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-/home/nfs06/yangs/ckpt/Qwen/Qwen3.5-4B/verl/mt/single_genrm}

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
# The repository's uv environment treats vLLM and SGLang as mutually exclusive.
# Override this in a custom image if mixed backends are installed together.
POLICY_ROLLOUT_BACKEND=${POLICY_ROLLOUT_BACKEND:-vllm}
REWARD_ROLLOUT_BACKEND=${REWARD_ROLLOUT_BACKEND:-vllm}
ENABLE_LANGUAGE_DETECTION=${ENABLE_LANGUAGE_DETECTION:-false}

REWARD_FUNCTION=${REPO_ROOT}/examples/rewards/mt_genrm_reward.py

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=512
    data.val_batch_size=256
    data.max_prompt_length=1280
    data.max_response_length=3072
    data.filter_overlong_prompts=True
    data.shuffle=True
    data.truncation=error
)

ACTOR=(
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.clip_ratio_low=0.0003
    actor_rollout_ref.actor.clip_ratio_high=0.0004
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${POLICY_ROLLOUT_BACKEND}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REWARD=(
    reward.num_workers=8
    reward.reward_manager.name=naive
    reward.reward_model.enable=True
    reward.reward_model.enable_resource_pool=False
    reward.reward_model.model_path="${REWARD_MODEL_PATH}"
    reward.reward_model.rollout.name="${REWARD_ROLLOUT_BACKEND}"
    reward.reward_model.rollout.tensor_model_parallel_size=1
    reward.reward_model.rollout.gpu_memory_utilization=0.6
    reward.reward_model.rollout.free_cache_engine=True
    reward.reward_model.rollout.skip_tokenizer_init=False
    reward.reward_model.rollout.max_num_batched_tokens=12000
    reward.reward_model.rollout.prompt_length=2048
    reward.reward_model.rollout.response_length=8192
    reward.custom_reward_function.path="${REWARD_FUNCTION}"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.model_name="${REWARD_MODEL_PATH}"
    +reward.custom_reward_function.reward_kwargs.extractor_type=codeblock
    +reward.custom_reward_function.reward_kwargs.score_scale_factor=0.01
    +reward.custom_reward_function.reward_kwargs.default_reward=-0.04
    +reward.custom_reward_function.reward_kwargs.enable_language_detection="${ENABLE_LANGUAGE_DETECTION}"
    +reward.custom_reward_function.reward_kwargs.max_prompt_length=2048
    +reward.custom_reward_function.reward_kwargs.max_tokens=8192
    +reward.custom_reward_function.reward_kwargs.temperature=0.6
    +reward.custom_reward_function.reward_kwargs.top_p=0.9
    +reward.custom_reward_function.reward_kwargs.top_k=-1
    +reward.custom_reward_function.reward_kwargs.overlong_buffer_enable=False
    +reward.custom_reward_function.reward_kwargs.max_response_length=3072
    +reward.custom_reward_function.reward_kwargs.overlong_buffer_length=2048
    +reward.custom_reward_function.reward_kwargs.overlong_penalty_factor=0.04
)

TRAINER=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=False
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.logger=['console','wandb']
    trainer.project_name=cot_mt_verl
    trainer.experiment_name=qwen3_5_mt_single_genrm
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=True
    trainer.log_val_generations=32
    trainer.save_freq=40
    trainer.test_freq=20
    trainer.resume_mode=auto
    trainer.total_epochs=1
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
)

cd "${REPO_ROOT}"
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
