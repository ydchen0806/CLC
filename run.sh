#!/bin/bash
# GRCL training example — single GPU
# Adjust paths to your environment before running.

python3 train_CLC.py \
    -m grcl \
    -d /path/to/flickr2w.hdf5 \
    --ref_path /path/to/flickr2k/ \
    --test_dataset /path/to/kodak.hdf5 \
    --feature_cache_path /path/to/feature_cache/flicker_features.pkl \
    --save_path ./checkpoints_grcl/ \
    --lambda 0.01 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --n_refs 3 \
    --N 128 \
    --n_clusters 3000 \
    --type mse \
    --patch-size 256 256 \
    --lr_epoch 30 40 \
    --cuda \
    --num-workers 4
