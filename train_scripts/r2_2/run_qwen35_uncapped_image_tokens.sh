#!/bin/bash
set -euo pipefail
set -x

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <experiment_id> <2b> <visual_prompt:true|false>" >&2
    exit 2
fi

EXP_NO=$1
MODEL_SIZE=$2
VISUAL_PROMPT=$3
ROOT=/mnt/shared-storage-user/mineru4s/jcwang/VPTrack
VP_COLOR=blue
VP_WIDTH=1
VP_LINE_STYLE=solid
VP_OPACITY=1.00

if [[ "$MODEL_SIZE" != "2b" ]]; then
        echo "Only the final Qwen3.5-2B model is retained" >&2
        exit 2
fi
MODEL=$ROOT/models/Qwen3.5-2B
MODEL_TAG=qwen35_2b

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
esac

cd "$ROOT"
source /mnt/shared-storage-user/mineru4s/jcwang/anaconda3/etc/profile.d/conda.sh
conda activate py12

export HF_HOME=/mnt/shared-storage-user/mineru4s/share/huggingface
export TIKTOKEN_CACHE_DIR=/mnt/shared-storage-user/mineru4s/jcwang/tiktoken_cache
export PATH=/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/mnt/shared-storage-user/mineru4s/share/library/cuda-12.8
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TOKENIZERS_PARALLELISM=false
export QWENVL_BBOX_FORMAT=new

EPOCHS=1
LR=2e-5
NNODES=1
GPUS=8
PDTB=4
GAS=4
BS=$((NNODES * GPUS * PDTB * GAS))

if [[ ! -s "$TRAIN_DATASET" ]]; then
    echo "Training data is missing: $TRAIN_DATASET" >&2
    exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Qwen3.5 model is not loadable: $MODEL" >&2
    exit 1
fi
records=$(wc -l < "$TRAIN_DATASET")
if [[ "$records" -ne 1000000 ]]; then
    echo "Training dataset has $records records; expected 1000000" >&2
    exit 1
fi

python -c 'from transformers import Qwen3_5ForConditionalGeneration; import qwen_vl_utils, fla, causal_conv1d, flash_attn' \
    || { echo "Qwen3.5 dependencies are incomplete" >&2; exit 1; }

export NNODES
export NPROC_PER_NODE=$GPUS
RUN_NAME=${EXP_NO}_${MODEL_TAG}_${RUN_TAG}_1m_lr${LR}_bs${BS}_e${EPOCHS}_pdbs${PDTB}_ga${GAS}
OUTPUT_DIR=$ROOT/outputs_vpt_r1/checkpoints/$RUN_NAME
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

# Do not set IMAGE_MAX_TOKEN_NUM: use the Qwen3.5 processor's uncapped default.
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
    --seed 42 \
    --data_seed 42 \
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
        --vp_color "$VP_COLOR"
        --vp_width "$VP_WIDTH"
        --vp_line_style "$VP_LINE_STYLE"
        --vp_opacity "$VP_OPACITY"
        --vp_placement previous_prediction
    )
fi
NPROC_PER_NODE=8 bash "$ROOT/test_scripts/infer_after_train_vpt.sh" \
    "$OUTPUT_DIR" "${RUN_NAME}_pool" "$VISUAL_PROMPT" "${INFER_ARGS[@]}"
