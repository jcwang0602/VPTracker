#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/ms-swift:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export QWENVL_BBOX_FORMAT=new
export NPROC_PER_NODE="${GPUS:-1}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DATASET="${DATASET:-$ROOT/data_jsonlines/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/VPTracker}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
if (( NPROC_PER_NODE < 1 || PER_DEVICE_BATCH_SIZE < 1 || BATCH_SIZE < 1 || BATCH_SIZE % (NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE) != 0 )); then
    echo 'BATCH_SIZE must be a positive multiple of GPUS * PER_DEVICE_BATCH_SIZE' >&2
    exit 2
fi
if [[ ! -s "$DATASET" ]]; then
    echo "Training JSONL is missing or empty: $DATASET (run data_preparation.sh first)" >&2
    exit 2
fi
# Preserve the model processor's default image resolution.
unset IMAGE_MAX_TOKEN_NUM
exec python -m swift.cli.main sft \
    --external_plugins "$ROOT/ms-swift/swift/template/vptracker_plugin.py" \
    --model "$MODEL" --model_type qwen3_5 --template vptracker \
    --dataset "$DATASET" --split_dataset_ratio 0 --eval_strategy no \
    --num_train_epochs 1 --learning_rate 2e-5 --warmup_ratio 0.05 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$((BATCH_SIZE / NPROC_PER_NODE / PER_DEVICE_BATCH_SIZE))" \
    --seed 42 --data_seed 42 --tuner_type full --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --freeze_vit false --freeze_aligner false \
    --gradient_checkpointing false --vit_gradient_checkpointing false \
    --enable_thinking false --add_non_thinking_prefix true --norm_bbox none \
    --max_length 4096 --lazy_tokenize true --remove_unused_columns false \
    --load_from_cache_file false --dataset_num_proc 4 --dataloader_num_workers 4 \
    --save_strategy epoch --save_only_model true --save_total_limit 1 \
    --logging_steps 10 --use_hf true --output_dir "$OUTPUT_DIR" "$@"
