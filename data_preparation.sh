# 01 tnllt train 150
sudo python dataset/sampling_dataset.py \
    --dataset tnl2k,tnllt \
    --dataset_ratio 7,3  \
    --sample_num 1000000 \
    --save_path save_path.jsonl \
    --phase train \
    --visual_prompt True \
    --vp_inbbox_ratio 0.9
# 02 tnllt val 50
sudo python dataset/sampling_dataset.py \
    --dataset tnl2k,tnllt \
    --dataset_ratio 7,3  \
    --sample_num 10000 \
    --save_path save_path.jsonl \
    --phase test \
    --visual_prompt True \
    --vp_inbbox_ratio 0.9

