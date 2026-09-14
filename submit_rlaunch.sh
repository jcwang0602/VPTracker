rlaunch --memory=320000 --cpu=48 \
    --charged-group=mineru4sh_gpu --private-machine=yes \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --mount=gpfs://gpfs1/wangjingchao:/mnt/shared-storage-user/wangjingchao \
    -- sudo bash /mnt/shared-storage-user/mineru4s/jcwang/VPTrack/dataset/gen_datasets_vlt_vp_ib09.sh
