import argparse
import math
import random
import sys
import time
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 表示第一张卡

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloader_ref_cluster import LICDataset
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim
import torch.multiprocessing as mp
from models import TCM, CLC
from torch.utils.tensorboard import SummaryWriter   

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    # optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(parameters)),
    #     lr=args.learning_rate,
    # )
    # aux_optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(aux_parameters)),
    #     lr=args.aux_learning_rate,
    # )
    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    
    return optimizer, aux_optimizer

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, scaler, use_mixed_precision, type='mse', args = None
):
    model.train()
    device = next(model.parameters()).device
    
    # 初始化计时器
    timers = defaultdict(float)
    
    for i, d in enumerate(train_dataloader):
        iter_start = time.time()
        
        # 数据加载计时
        data_start = time.time()
        sample, ref_samples, key, ref_keys = d
        sample = sample.to(device)
        ref_samples = [r.to(device) for r in ref_samples]
        timers['data_loading'] += time.time() - data_start

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 前向传播计时
        forward_start = time.time()
        if use_mixed_precision:
            with torch.amp.autocast('cuda'):
                if args.model == 'clc':
                    out_net = model(sample, ref_samples)
                else:
                    out_net = model(sample)
                # out_net = model(sample)
                out_criterion = criterion(out_net, sample)
            scaler.scale(out_criterion["loss"]).backward()
        else:
            if args.model == 'clc':
                out_net = model(sample, ref_samples)
            else:
                out_net = model(sample)

            out_criterion = criterion(out_net, sample)
            out_criterion["loss"].backward()
        timers['forward_pass'] += time.time() - forward_start

        # 反向传播和优化计时
        backward_start = time.time()
        if clip_max_norm > 0:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        if use_mixed_precision:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_()
            scaler.step(optimizer)
            scaler.update()
        else:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_()
            optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        timers['backward_pass'] += time.time() - backward_start

        if i % 500 == 0:
            # 计算每次迭代的总时间
            iter_time = time.time() - iter_start
            timers['total'] += iter_time
            
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(sample)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.5f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                    f"\tAux loss: {aux_loss.item():.3f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(sample)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.5f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.5f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                    f"\tAux loss: {aux_loss.item():.3f}"
                )
            
            # 打印每个部分的耗时统计
            print(f"Timing stats (sec) - "
                  f"Data loading: {timers['data_loading']/(i+1):.3f}/iter | "
                  f"Forward pass: {timers['forward_pass']/(i+1):.3f}/iter | "
                  f"Backward pass: {timers['backward_pass']/(i+1):.3f}/iter | "
                  f"Total: {timers['total']/(i+1):.3f}/iter")

def test_epoch(epoch, test_dataloader, model, criterion, type='mse', args = None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            sample, ref_samples, key, ref_keys = d
            sample = sample.to(device)
            ref_samples = [r.to(device) for r in ref_samples]
            if args.model == 'clc':
                out_net = model(sample, ref_samples)
            else:
                out_net = model(sample)
            out_criterion = criterion(out_net, sample)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if type == 'mse':
                mse_loss.update(out_criterion["mse_loss"])
            else:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

    if type == 'mse':
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.5f} |"
            f"\tMSE loss: {mse_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.3f}\n"
        )
    else:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.5f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.3f}\n"
        )

    return loss.avg

def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="clc",
        choices=["tcm", "clc"],
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--test_dataset", type=str, default="/h3cstore_ns/ydchen/DATASET/kodak.hdf5", help="Test dataset")
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1,
        type=float,
        help="gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    parser.add_argument("--ref_path", type=str, help="Path to reference dataset for CLC")
    parser.add_argument("--feature_cache_path", type=str, help="Path to feature cache for CLC")
    parser.add_argument("--n_clusters", type=int, default=3000, help="Number of clusters for CLC")
    parser.add_argument("--n_refs", type=int, default=3, help="Number of reference images for CLC")

    # New argument for mixed precision training
    parser.add_argument(
        "--use-mixed-precision",
        default=False,
        type=bool,
        help="Use mixed precision training (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    if args.model == "clc" or args.model == "tcm":
        train_dataset = LICDataset(
            args.dataset,
            args.ref_path,
            transform=train_transforms,
            feature_cache_path=args.feature_cache_path,
            n_clusters=args.n_clusters,
            n_refs=args.n_refs
        )
        test_dataset = LICDataset(
            args.test_dataset,
            args.ref_path,
            transform=test_transforms,
            feature_cache_path=args.feature_cache_path,
            n_clusters=args.n_clusters,
            n_refs=args.n_refs
        )
    else:
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if args.model == "clc":
        net = CLC(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
    else:
        net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)


    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        new_weights = {}
        for k, v in checkpoint['state_dict'].items():
            new_weights[k.replace('module.', '')] = v
        net.load_state_dict(new_weights)
        # if args.continue_train:
        #     last_epoch = checkpoint["epoch"] + 1
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    best_loss = float("inf")

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            scaler,
            args.use_mixed_precision,
            type,
            args = args
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type, args = args)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(sys.argv[1:])