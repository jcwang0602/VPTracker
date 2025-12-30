CHECKPOINT="Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")
GPUS=$(nvidia-smi -L | wc -l)

export BNB_CUDA_VERSION=124
export PYTHONPATH="$(pwd):${PYTHONPATH}"

for (( i=0; i<${GPUS}; i++ ))
do
    export CUDA_VISIBLE_DEVICES=$i
    export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
    CUDA_VISIBLE_DEVICES=$i python evaluation/infer_tracking_qwen_vlt_crop_absent.py \
        --checkpoint ${CHECKPOINT} \
        --model_name ${MODEL_NAME} \
        --dataset tnllt \
        --save_dir outputs/results \
        --visual_prompt \
        --vp_scale 4 \
        --infer_backend vllm \
        --seg_index $i \
        --seg_total ${GPUS}  >> ${CHECKPOINT}/eval_log_${i}.txt 2>&1 &
done

wait
