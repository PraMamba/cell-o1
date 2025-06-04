#!/bin/bash
set -x

# ------------------------ Configurable Parameters ------------------------

# Path to preprocessed training and validation datasets (in Parquet format)
TRAIN_DATA="/path/to/train.parquet"
VAL_DATA="/path/to/test.parquet"

# Path to the merged SFT checkpoint (after merge_and_unload)
SFT_MERGED_CKPT="Qwen/Qwen2.5-7B-Instruct"

# Path to your custom reward function
REWARD_FN="verl/workers/reward_function/compute_score.py"

# Logging and experiment tracking (optional with wandb)
PROJECT_NAME="your_project_name"
EXPERIMENT_NAME="your_experiment_name"

# ------------------------ GRPO Training Launch ------------------------

# Enable more efficient attention implementation
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    critic.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path=${SFT_MERGED_CKPT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.enable=false \
    +reward.reward_manager=custom \
    reward_model.micro_batch_size_per_gpu=1 \
    custom_reward_function.path=${REWARD_FN} \
    trainer.critic_warmup=0 \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=20 2>&1 | tee grpo_training.log
