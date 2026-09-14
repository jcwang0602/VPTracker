#!/bin/bash
set -euo pipefail
set -x

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <experiment_id> <seed> <visual_prompt:true|false> [data_seed]" >&2
    exit 2
fi

EXP_NO=$1
SEED=$2
VISUAL_PROMPT=$3
DATA_SEED=${4:-$SEED}
ROOT=/mnt/shared-storage-user/mineru4s/jcwang/VPTrack
MODEL=$ROOT/models/Qwen3.5-2B
MODEL_TAG=qwen35_2b

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "seed must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$DATA_SEED" =~ ^[0-9]+$ ]]; then
    echo "data_seed must be a non-negative integer" >&2
    exit 2
fi

case "$VISUAL_PROMPT" in
    true)
        TRAIN_DATASET=$ROOT/data_vpt/tt_73_vlt_train_1m_ib075_vp_blue_w1_opacity100.jsonl
        RUN_TAG=vp_blue_w1_op100_out25_pixelcoords_no_imgtok_cap
        ;;
    false)
        TRAIN_DATASET=$ROOT/data_vpt/tt_73_vlt_train_1m.jsonl
        RUN_TAG=no_vp_pixelcoords_no_imgtok_cap
        ;;
    *)
        echo "visual_prompt must be true or false" >&2
        exit 2
        ;;
esac

cd "$ROOT"
source /mnt/shared-storage-user/mineru4s/jcwang/anaconda3/etc/profile.d/conda.sh
conda activate py12

export HF_HOME=/mnt/shared-storage-user/mineru4s/share/huggingface
export TIKTOKEN_CACHE_DIR=/mnt/shared-storage-user/mineru4s/jcwang/tiktoken_cache
export PATH=/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8
export TOKENIZERS_PARALLELISM=false
export QWENVL_BBOX_FORMAT=new

EPOCHS=1
LR=2e-5
GPUS=8
PDTB=4
GAS=4
BS=$((GPUS * PDTB * GAS))

if [[ ! -s "$TRAIN_DATASET" || ! -f "$MODEL/config.json" ]]; then
    echo "Training data or Qwen3.5-2B model is unavailable" >&2
    exit 1
fi

SEED_TAG=seed${SEED}
if [[ $# -eq 4 ]]; then
    SEED_TAG=${SEED_TAG}_dataseed${DATA_SEED}
fi
RUN_NAME=${EXP_NO}_${MODEL_TAG}_${RUN_TAG}_${SEED_TAG}_1m_lr${LR}_bs${BS}_e${EPOCHS}_pdbs${PDTB}_ga${GAS}
OUTPUT_DIR=$ROOT/outputs_vpt_r1/checkpoints/$RUN_NAME
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=$GPUS \
swift sft \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --dataset "$TRAIN_DATASET" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PDTB" \
    --learning_rate "$LR" \
    --gradient_accumulation_steps "$GAS" \
    --seed "$SEED" \
    --data_seed "$DATA_SEED" \
    --load_from_cache_file false \
    --remove_unused_columns false \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
    --add_non_thinking_prefix true \
    --eval_strategy no \
    --save_strategy epoch \
    --save_only_model true \
    --save_total_limit 1 \
    --logging_steps 10 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --use_hf true \
    --dataset_num_proc 16 \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 8 \
    >> "$OUTPUT_DIR/log.txt" 2>&1

INFER_ARGS=(--model_type qwen3_5 --stable_sharding)
if [[ "$VISUAL_PROMPT" == "true" ]]; then
    INFER_ARGS+=(
        --vp_color blue
        --vp_width 1
        --vp_line_style solid
        --vp_opacity 1.0
        --vp_placement previous_prediction
    )
fi

NPROC_PER_NODE=8 bash "$ROOT/test_scripts/infer_after_train_vpt.sh" \
    "$OUTPUT_DIR" "${RUN_NAME}_pool" "$VISUAL_PROMPT" "${INFER_ARGS[@]}"
