python3 train_CLC.py \
    -d /img_video/img/Flicker2W.hdf5 \
    --ref_path /img_video/img/Flicker2K.hdf5 \
    --feature_cache_path /h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/flicker_features.pkl \
    --save_path /h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/trained_model_1021 \ 
    --lambda 0.01 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --n_refs 3 \
    --n_clusters 1000 \
    --type mse \
    --patch-size 256 256 \
    --cuda \
    --num-workers 0


    #     parser.add_argument('--data_path', type=str, default='/img_video/img/Flicker2W.hdf5', help='Path to the main dataset')
    # parser.add_argument('--ref_path', type=str, default='/img_video/img/Flicker2K.hdf5', help='Path to the reference dataset')
    # parser.add_argument('--feature_cache_path', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/flicker_features.pkl', help='Path to feature cache')
    # parser.add_argument('--output_base_dir', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/comparison_results', help='Base output directory for results')
    # parser.add_argument('--n_clusters', type=int, default=1000, help='Number of clusters')
    # parser.add_argument('--n_refs', type=int, default=3, help='Number of reference images')
    # parser.add_argument('--num_comparisons', type=int, default=10, help='Number of comparisons to visualize')