MODEL=Qwen/Qwen3-VL-4B-Instruct
LR=2e-5
EPOCHS=1
GPUS=8
BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export NPROC_PER_NODE=${GPUS}
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

DATASET=data/train.jsonl
VAL_DATASET=data/val.jsonl
OUTPUT_DIR=outputs/checkpoints/VPTracker
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

# 2 * 21GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=8 \
swift sft \
    --model $MODEL \
    --model_type='qwen3_vl' \
    --new_special_tokens 'data/tokens.txt' \
    --dataset $DATASET \
    --val_dataset $VAL_DATASET \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --learning_rate $LR \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --load_from_cache_file False \
    --remove_unused_columns False \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --use_hf true \
    --dataset_num_proc 16 \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 8  >> ${OUTPUT_DIR}/log.txt 2>&1