export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0   # 设定为你的网络接口名
export NCCL_IB_DISABLE=1         # 如果不使用 InfiniBand，可以禁用它
export NCCL_P2P_DISABLE=1
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_ASYNC_ERROR_HANDLING=1
python3 -m torch.distributed.run --nproc_per_node=8 train_CLC_ddp.py \
    -d /img_video/img/Flicker2W.hdf5 \
    --ref_path /img_video/img/Flicker2K.hdf5 \
    --feature_cache_path /h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/flicker_features.pkl \
    --save_path /h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/trained_model_1021_ddp \
    --lambda 0.01 \
    --epochs 50 \
    --batch-size 12 \
    --learning-rate 1e-4 \
    --n_refs 3 \
    --n_clusters 1000 \
    --type mse \
    --patch-size 256 256 \
    --cuda \
    --num-workers 0