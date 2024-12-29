import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM, CLC
from dataloader_ref_cluster import LICDataset
import warnings
import os
import sys
import math
import argparse
import time
from pytorch_msssim import ms_ssim
from PIL import Image

warnings.filterwarnings("ignore")

# print(torch.cuda.is_available())

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default='/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/tcm_trained_model_final_modify_no_amp/0.0035checkpoint_best.pth.tar',
                        help="Path to a checkpoint")
    parser.add_argument("--data", type=str, default='/h3cstore_ns/ydchen/DATASET/kodak.hdf5',
                        help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.add_argument(
        "--model", type=str, choices=['tcm', 'clc'], default='tcm', help="Model to use"
    )
    parser.add_argument("--ref_path", type=str, default='/h3cstore_ns/ydchen/DATASET/coding_img_cropped_2/Flickr2K.hdf5',
                        help="Path to reference dataset for CLC")
    parser.add_argument("--feature_cache_path", type=str, default='/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/model_ckpt_TCM/data_cluster_feature/flicker_features.pkl',
                        help="Path to feature cache for CLC")
    parser.add_argument("--n_clusters", type=int, default=3000, help="Number of clusters for CLC")
    parser.add_argument("--n_refs", type=int, default=3, help="Number of reference images for CLC")
    parser.set_defaults(real=False)
    args = parser.parse_args()
    return args

def main(args):
    p = 128
    path = args.data
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'


    net = CLC(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    test_dataset = LICDataset(
        args.data,
        args.ref_path,
        transform=transforms.ToTensor(),
        feature_cache_path=args.feature_cache_path,
        n_clusters=args.n_clusters,
        n_refs=args.n_refs
    )

    net = net.to(device)
    net.eval()

    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)


    net.update()

    for sample, ref_samples, key, ref_keys in test_dataset:
        x = sample.unsqueeze(0).to(device)
        ref_samples = [r.unsqueeze(0).to(device) for r in ref_samples]
        x_padded, padding = pad(x, p)
        ref_samples_padded = [pad(r, p)[0] for r in ref_samples]
        count += 1
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            s = time.time()
            if args.model == 'clc':
                out_enc = net.compress(x_padded, ref_samples_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"], ref_samples_padded, x.shape)
            else:
                out_enc = net.compress(x_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            total_time += (e - s)
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            print(f'{key} Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
            print(f'{key} PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
            Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            PSNR += compute_psnr(x, out_dec["x_hat"])




    PSNR = PSNR / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    args = parse_args()
    main(args)