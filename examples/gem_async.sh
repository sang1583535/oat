GAME_ID="game:Sokoban-v0-easy"
MODEL_ID="Qwen/Qwen3-4B-Base"
GPUS=4
GRADIENT_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE_PER_DEVICE=$((GRADIENT_BATCH_SIZE / GPUS))
PI_BUFFER_MAXLEN_PER_DEVICE=$((GRADIENT_BATCH_SIZE / GPUS))
NUM_ENV=8

export VLLM_USE_V1=1
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

python -m oat.experiment.run_gem_async \
    --env_id $GAME_ID \
    --num_env $NUM_ENV \
    --prompt_template qwen3_game \
    --wrappers concat \
    --gamma 0.9 \
    --norm_return \
    --gpus $GPUS \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle $GAME_ID \
    --pretrain $MODEL_ID \
    --num_samples 1 \
    --rollout_batch_size $GRADIENT_BATCH_SIZE \
    --rollout_batch_size_per_device $ROLLOUT_BATCH_SIZE_PER_DEVICE \
    --pi_buffer_maxlen_per_device $PI_BUFFER_MAXLEN_PER_DEVICE \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.65 \
    --zero-stage 2 \
    --max-train 999999 \
    --generate_max_length 4096 \
    --max_model_len 16384 \
    --eval_generate_max_length 4096 \
    --eval_steps -1 \
    --use-wb \
    --wb-run-name AsyncRL-${MODEL_ID}-${GAME_ID} \
    "$@"
