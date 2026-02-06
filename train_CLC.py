"""Training script for CLC / GRCL image compression models."""

import argparse, math, random, sys, os, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=FutureWarning)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ms_ssim
import torch.multiprocessing as mp

from dataloader_ref_cluster import LICDataset
from models import CLC, GRCL

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Loss ──────────────────────────────────────────────────────────────────────
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, metric='mse'):
        super().__init__()
        self.lmbda = lmbda; self.metric = metric; self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size(); num_px = N * H * W
        bpp = sum(torch.log(lh).sum() / (-math.log(2) * num_px) for lh in output["likelihoods"].values())
        out = {"bpp_loss": bpp}
        if self.metric == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + bpp
        else:
            out["ms_ssim_loss"] = ms_ssim(output["x_hat"], target, data_range=1.)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + bpp
        return out

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self, v, n=1): self.val=v; self.sum+=v*n; self.count+=n; self.avg=self.sum/self.count

class CustomDP(nn.DataParallel):
    def __getattr__(self, k):
        try: return super().__getattr__(k)
        except AttributeError: return getattr(self.module, k)

# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, criterion, loader, opt, aux_opt, epoch, clip, metric, use_ref):
    model.train(); dev = next(model.parameters()).device; t = defaultdict(float)
    for i, (sample, refs, *_) in enumerate(loader):
        sample = sample.to(dev); refs = [r.to(dev) for r in refs]
        opt.zero_grad(); aux_opt.zero_grad()
        out = model(sample, refs) if use_ref else model(sample)
        loss = criterion(out, sample)
        loss["loss"].backward()
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            if p.grad is not None: p.grad.nan_to_num_()
        opt.step()
        aux = model.aux_loss(); aux.backward(); aux_opt.step()
        if i % 500 == 0:
            dist_key = "mse_loss" if metric == 'mse' else "ms_ssim_loss"
            print(f"  [{epoch}] {i}/{len(loader)} | loss={loss['loss'].item():.5f} "
                  f"| {dist_key}={loss[dist_key].item():.5f} | bpp={loss['bpp_loss'].item():.3f} | aux={aux.item():.3f}")

@torch.no_grad()
def test_epoch(epoch, loader, model, criterion, metric, use_ref):
    model.eval(); dev = next(model.parameters()).device
    loss_m, bpp_m, dist_m = AverageMeter(), AverageMeter(), AverageMeter()
    for sample, refs, *_ in loader:
        sample = sample.to(dev); refs = [r.to(dev) for r in refs]
        out = model(sample, refs) if use_ref else model(sample)
        c = criterion(out, sample)
        loss_m.update(c["loss"]); bpp_m.update(c["bpp_loss"])
        dist_m.update(c.get("mse_loss", c.get("ms_ssim_loss", 0)))
    dk = "MSE" if metric == 'mse' else "MS-SSIM"
    print(f"  Test [{epoch}] loss={loss_m.avg:.5f} | {dk}={dist_m.avg:.5f} | bpp={bpp_m.avg:.3f}")
    return loss_m.avg

def save_ckpt(state, is_best, epoch, path):
    torch.save(state, os.path.join(path, "ckpt_latest.pth.tar"))
    if epoch % 5 == 0: torch.save(state, os.path.join(path, f"ckpt_ep{epoch}.pth.tar"))
    if is_best: torch.save(state, os.path.join(path, "ckpt_best.pth.tar"))

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="grcl", choices=["clc", "grcl"])
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("--test_dataset", type=str, default="")
    p.add_argument("--ref_path", type=str, required=True)
    p.add_argument("--feature_cache_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("-e", "--epochs", type=int, default=50)
    p.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    p.add_argument("--aux-learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-batch-size", type=int, default=1)
    p.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
    p.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--n_clusters", type=int, default=3000)
    p.add_argument("--n_refs", type=int, default=3)
    p.add_argument("--lr_epoch", nargs='+', type=int, default=[30, 40])
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    save_path = os.path.join(args.save_path, str(args.lmbda))
    os.makedirs(save_path, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)

    tr_tf = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    te_tf = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    make_ds = lambda p, tf: LICDataset(p, args.ref_path, transform=tf,
        feature_cache_path=args.feature_cache_path, n_clusters=args.n_clusters, n_refs=args.n_refs)
    train_ds = make_ds(args.dataset, tr_tf)
    test_ds = make_ds(args.test_dataset or args.dataset, te_tf)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    tr_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(test_ds, args.test_batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.model == 'grcl':
        net = GRCL(N=args.N, M=320, num_slices=5, num_ref=args.n_refs, clm_type='simple')
    else:
        net = CLC(N=args.N, M=320, num_slices=5, max_support_slices=5, num_ref_frames=args.n_refs)
    use_ref = True

    params = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_p = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    pd = dict(net.named_parameters())
    opt = optim.AdamW((pd[n] for n in sorted(params)), lr=args.learning_rate)
    aux_opt = optim.AdamW((pd[n] for n in sorted(aux_p)), lr=args.aux_learning_rate)
    sched = optim.lr_scheduler.MultiStepLR(opt, args.lr_epoch, gamma=0.1)
    criterion = RateDistortionLoss(args.lmbda, args.type)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1: net = CustomDP(net)

    best = float('inf')
    for ep in range(args.epochs):
        print(f"Epoch {ep}, lr={opt.param_groups[0]['lr']:.6f}")
        train_one_epoch(net, criterion, tr_dl, opt, aux_opt, ep, args.clip_max_norm, args.type, use_ref)
        loss = test_epoch(ep, te_dl, net, criterion, args.type, use_ref)
        sched.step()
        is_best = loss < best; best = min(loss, best)
        save_ckpt({'epoch': ep, 'state_dict': net.state_dict(), 'loss': loss,
                   'optimizer': opt.state_dict(), 'aux_optimizer': aux_opt.state_dict()},
                  is_best, ep, save_path)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main(sys.argv[1:])
