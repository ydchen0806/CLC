# Conditional Latent Coding (CLC) for Deep Image Compression

This repository provides the official PyTorch implementation of the paper:

> **Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression**  
> *Siqi Wu†, Yinda Chen†, Dong Liu, Zhihai He*
>
> † Equal contribution

## Introduction

Conditional Latent Coding (CLC) is a deep image compression framework that leverages conditional coding with learnable synthesized references to achieve efficient compression. This method is built upon [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the TCM framework.

In this repository, we provide the code for:

- The CLC model (`CLC_run.py`)
- Data loader that supports reference clustering (`dataloader_ref_cluster.py`)
- Training script for CLC (`train_CLC.py`)

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- scikit-learn
- h5py
- matplotlib
- tqdm
- pillow
- tensorboard
- compressai
- einops
- timm

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/CLC.git
   cd CLC
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can set up a conda environment:

   ```bash
   conda create -n clc_env python=3.8
   conda activate clc_env

   # Install PyTorch (modify according to your CUDA version)
   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

   # Install other dependencies
   pip install numpy scikit-learn h5py matplotlib tqdm pillow tensorboard compressai einops timm
   ```

3. **Docker**: A Docker image is available for this project. You can pull it using
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/dockerhub1913/mamba0224_ydchen:latest
  ```
## Data Preparation

The CLC model requires datasets for training and reference images.

### Main Dataset

Prepare your main dataset in HDF5 format. The dataset should contain images stored in HDF5 datasets.

Example structure:

```
/path/to/your_dataset.h5
    ├── image_00001
    ├── image_00002
    ├── ...
```

### Reference Dataset

Prepare a reference dataset, either in HDF5 format or as a directory of images. The reference images will be used for conditional coding.

Example structure for HDF5:

```
/path/to/reference_dataset.h5
    ├── ref_image_0001
    ├── ref_image_0002
    ├── ...
```

Or for a directory:

```
/path/to/reference_images/
    ├── ref_image_0001.jpg
    ├── ref_image_0002.png
    ├── ...
```

### Precomputing Features

For efficient reference selection, precompute features for the reference dataset using the provided script `dataloader_ref_cluster.py`.

Example:

```bash
python dataloader_ref_cluster.py \
    --data_path /path/to/your_dataset.h5 \
    --ref_path /path/to/reference_dataset.h5 \
    --feature_cache_path /path/to/save/feature_cache.pkl \
    --output_base_dir /path/to/save/comparison_results \
    --n_clusters 1000 \
    --n_refs 3 \
    --num_comparisons 10
```

- `--data_path`: Path to your main dataset.
- `--ref_path`: Path to your reference dataset.
- `--feature_cache_path`: Path to save the computed features.
- `--output_base_dir`: Directory to save visualization results.
- `--n_clusters`: Number of clusters to create for the reference images.
- `--n_refs`: Number of reference images to use during training.
- `--num_comparisons`: Number of sample comparisons to visualize.

## Usage

### Training

To train the CLC model, use the `train_CLC.py` script.

Example:

```bash
python train_CLC.py \
    -d /path/to/your_dataset.h5 \
    --ref_path /path/to/reference_dataset.h5 \
    --feature_cache_path /path/to/feature_cache.pkl \
    --save_path /path/to/save/checkpoints/ \
    --lambda 0.01 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --n_refs 3 \
    --n_clusters 1000 \
    --type mse \
    --patch-size 256 256 \
    --cuda \
    --num-workers 4
```

Explanation of the arguments:

- `-d`, `--dataset`: Path to the main dataset.
- `--ref_path`: Path to the reference dataset.
- `--feature_cache_path`: Path to the precomputed feature cache.
- `--save_path`: Directory to save model checkpoints and logs.
- `--lambda`: Rate-distortion tradeoff parameter.
- `--epochs`: Number of training epochs.
- `--batch-size`: Batch size for training.
- `--learning-rate`: Learning rate.
- `--n_refs`: Number of reference images to use.
- `--n_clusters`: Number of clusters for reference images.
- `--type`: Loss type (`mse` or `ms-ssim`).
- `--patch-size`: Size of image patches for training.
- `--cuda`: Use CUDA for training.
- `--num-workers`: Number of data loading workers.

**Notes**:

- The code supports both MSE and MS-SSIM loss functions.
- The `--save_path` directory will contain checkpoints and tensorboard logs.
- Adjust `--lambda` to trade off between bit-rate and distortion.

### Evaluation

To evaluate the trained model, modify the `train_CLC.py` script or create a new evaluation script. An evaluation script will be provided in future updates.

## Pretrained Models

Pretrained models will be provided in future updates.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@article{wu2023conditional,
  title={Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression},
  author={Wu, Siqi and Chen, Yinda and Liu, Dong and He, Zhihai},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## Acknowledgements

This code is built upon [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the [TCM framework](https://github.com/jmliu206/LIC_TCM). We thank the authors for their contributions to the community.

## License

This project is licensed under the terms of the MIT license.

## Contact

For questions or comments, please open an issue on this repository or contact us at cyd0806@mail.ustc.edu.cn.


