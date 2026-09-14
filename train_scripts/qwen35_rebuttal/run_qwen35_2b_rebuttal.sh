#!/bin/bash
set -euo pipefail
set -x

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <experiment_id> <finetune|r22|seed_vp|seed_no_vp> <variant> [seed]" >&2
    exit 2
fi

EXP_NO=$1
GROUP=$2
VARIANT=$3
SEED=${4:-42}
ROOT=/mnt/shared-storage-user/mineru4s/jcwang/VPTrack
MODEL=$ROOT/models/Qwen3.5-2B
MODEL_TAG=qwen35_2b
VP_ENABLED=false
VP_COLOR=blue
VP_WIDTH=1
VP_STYLE=solid
VP_OPACITY=1.00
TUNER_ARGS=(--tuner_type full --freeze_vit false --freeze_aligner false)

case "$GROUP" in
    finetune)
        TRAIN_DATASET=$ROOT/data_vpt/tt_73_vlt_train_1m_ib075_vp_blue_w1_opacity100.jsonl
        VP_ENABLED=true
        case "$VARIANT" in
            lora)
                TUNER_ARGS=(
                    --tuner_type lora
                    --freeze_vit true
                    --freeze_aligner true
                    --target_modules all-linear
                    --lora_rank 64
                    --lora_alpha 128
                    --lora_dropout 0.05
                )
                ;;
            freeze_vit)
                TUNER_ARGS=(--tuner_type full --freeze_vit true --freeze_aligner false)
                ;;
            llm_only)
                TUNER_ARGS=(--tuner_type full --freeze_vit true --freeze_aligner true)
                ;;
            *) echo "Unknown finetune variant: $VARIANT" >&2; exit 2 ;;
        esac
        RUN_TAG=vp_blue_w1_out25_${VARIANT}_pixelcoords_no_imgtok_cap
        ;;
    r22)
        TRAIN_DATASET=$ROOT/data_vpt/qwen35_r22_ablation/${VARIANT}.jsonl
        VP_ENABLED=true
        case "$VARIANT" in
            out00) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            out10) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            out50) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            red) VP_COLOR=red; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            green) VP_COLOR=green; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            width3) VP_COLOR=blue; VP_WIDTH=3; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            width5) VP_COLOR=blue; VP_WIDTH=5; VP_STYLE=solid; VP_OPACITY=1.00 ;;
            dashed) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=dashed; VP_OPACITY=1.00 ;;
            opacity025) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=0.25 ;;
            opacity050) VP_COLOR=blue; VP_WIDTH=1; VP_STYLE=solid; VP_OPACITY=0.50 ;;
            *) echo "Unknown R2.2 variant: $VARIANT" >&2; exit 2 ;;
        esac
        RUN_TAG=vp_${VARIANT}_pixelcoords_no_imgtok_cap
        ;;
    seed_vp)
        TRAIN_DATASET=$ROOT/data_vpt/tt_73_vlt_train_1m_ib075_vp_blue_w1_opacity100.jsonl
        VP_ENABLED=true
        RUN_TAG=vp_blue_w1_out25_seed${SEED}_pixelcoords_no_imgtok_cap
        ;;
    seed_no_vp)
        TRAIN_DATASET=$ROOT/data_vpt/tt_73_vlt_train_1m.jsonl
        RUN_TAG=no_vp_seed${SEED}_pixelcoords_no_imgtok_cap
        ;;
    *) echo "Unknown group: $GROUP" >&2; exit 2 ;;
esac

if [[ ! -s "$TRAIN_DATASET" || ! -f "$MODEL/config.json" ]]; then
    echo "Training dataset or Qwen3.5-2B model is unavailable" >&2
    exit 1
fi
if [[ $(wc -l < "$TRAIN_DATASET") -ne 1000000 ]]; then
    echo "Expected a 1M-record training dataset: $TRAIN_DATASET" >&2
    exit 1
fi

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
GPUS=8
PDTB=4
GAS=4
BS=$((GPUS * PDTB * GAS))
RUN_NAME=${EXP_NO}_${MODEL_TAG}_${RUN_TAG}_1m_lr${LR}_bs${BS}_e${EPOCHS}_pdbs${PDTB}_ga${GAS}
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
    --data_seed "$SEED" \
    --load_from_cache_file false \
    --remove_unused_columns false \
    "${TUNER_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
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

if [[ "$VARIANT" == lora ]]; then
    CHECKPOINT=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' ! -name '*-merged' -print | sort -V | tail -n 1)
    swift export --adapters "$CHECKPOINT" --merge_lora true --output_dir "${CHECKPOINT}-merged" >> "$OUTPUT_DIR/merge_lora.log" 2>&1
fi

INFER_ARGS=(--model_type qwen3_5 --stable_sharding)
if [[ "$VP_ENABLED" == true ]]; then
    INFER_ARGS+=(--vp_color "$VP_COLOR" --vp_width "$VP_WIDTH" --vp_line_style "$VP_STYLE" --vp_opacity "$VP_OPACITY" --vp_placement previous_prediction)
fi
NPROC_PER_NODE=8 bash "$ROOT/test_scripts/infer_after_train_vpt.sh" \
    "$OUTPUT_DIR" "${RUN_NAME}_pool" "$VP_ENABLED" "${INFER_ARGS[@]}"
