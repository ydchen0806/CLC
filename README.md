# GRCL: Generative Reference of Conditional Latents for Learned Image Compression

Official PyTorch implementation of:

> **Learned Image Coding with Generative Reference of Conditional Latents**
> Siqi Wu†, Yinda Chen†, Weiming Chen, Dong Liu, K. C. Ho, and Zhihai He
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025*
> † Equal contribution

This repository extends our AAAI 2025 Oral paper [CLC](https://arxiv.org/abs/2502.09971) with a more generic framework that incorporates three reference generation methods and adaptive selection.

## Architecture

```
Input x ──→ g_a ──→ y ──┬──→ CLM ──→ y_a ──→ CLS ──→ y_f ──→ Entropy Coding ──→ Bitstream
                         │
References ──→ g_a ──→ Y_r ──┘                              ↑ ref_features
```

**Key modules:**
- **CLM** (Conditional Latent Matching): Similarity-based matching + multi-scale deformable alignment
- **CLS** (Conditional Latent Synthesis): Adaptive α-blending: `μ = α⊙y + (1-α)⊙y_a`
- **Three reference methods**: Local dictionary, web search, image-text-image generation
- **Adaptive selection**: RD-optimal method chosen per image

## Project Structure

```
├── models/
│   ├── GRCL.py               # Main GRCL model (TCM backbone + CLM + CLS + entropy)
│   ├── CLM.py                 # Conditional Latent Matching (full & simplified)
│   ├── CLS.py                 # Conditional Latent Synthesis
│   ├── CLC_run.py             # Legacy CLC model (AAAI version)
│   └── reference_generation.py # Web/ITI reference generation modules
├── train_CLC.py               # Training script (supports CLC and GRCL)
├── eval_CLC.py                # Evaluation with BD-rate computation
├── dataloader_ref_cluster.py  # Dataset with Ball-tree reference retrieval
└── run.sh / run_ddp.sh        # Launch scripts
```

## Requirements

```
torch >= 1.12
compressai >= 1.2
torchvision
einops
timm
pytorch-msssim
scikit-learn
h5py
pillow
```

## Quick Start

### Training

```bash
python train_CLC.py \
    -m grcl \
    -d /path/to/flickr2w.hdf5 \
    --ref_path /path/to/flickr2k_crops/ \
    --feature_cache_path /path/to/feature_cache/ \
    --save_path ./checkpoints/ \
    --lambda 0.01 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --n_refs 3 \
    --n_clusters 3000 \
    --type mse \
    --cuda
```

### Evaluation

```bash
python eval_CLC.py \
    --model grcl \
    --checkpoint ./checkpoints/0.01/ckpt_best.pth.tar \
    --data /path/to/kodak/ \
    --ref_dir /path/to/kodak_refs/ \
    --cuda
```

## Pretrained Models

Checkpoints available on [HuggingFace](https://huggingface.co/ydchen0806/CLC).

## Citation

```bibtex
@article{wu2025grcl,
  title={Learned Image Coding with Generative Reference of Conditional Latents},
  author={Wu, Siqi and Chen, Yinda and Chen, Weiming and Liu, Dong and Ho, K. C. and He, Zhihai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}

@inproceedings{wu2025conditional,
  title={Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression},
  author={Wu, Siqi and Chen, Yinda and Liu, Dong and He, Zhihai},
  booktitle={AAAI},
  year={2025}
}
```

## Acknowledgements

Built upon [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the [TCM](https://github.com/jmliu206/LIC_TCM) framework.

## License

MIT License. Contact: cyd0806@mail.ustc.edu.cn
