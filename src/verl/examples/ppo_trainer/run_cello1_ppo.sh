#!/bin/bash
# Disclaimer: The model used in this script is for academic research purposes only.

set -x

# ------------------------ Configurable Parameters ------------------------

# Path to your training and validation datasets (Parquet format)
TRAIN_DATA="/path/to/train.parquet"
VAL_DATA="/path/to/test.parquet"

# Path to merged SFT checkpoint (used for both actor and critic initialization)
MERGED_CKPT="Qwen/Qwen2.5-7B-Instruct"

# Path to your custom reward function
REWARD_FN="verl/workers/reward_function/compute_score.py"

# WandB configuration (optional)
PROJECT_NAME="your_project_name"
EXPERIMENT_NAME="your_experiment_name"

# ------------------------ PPO/GRPO Training Launch ------------------------

# Use xFormers attention backend (recommended for vLLM + Qwen)
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MERGED_CKPT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=${MERGED_CKPT} \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=5 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.enable=false \
    +reward.reward_manager=custom \
    custom_reward_function.path=${REWARD_FN} \
    reward_model.micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger='[console, wandb]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME}
