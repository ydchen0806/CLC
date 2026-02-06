"""Evaluation script for CLC / GRCL with BD-rate computation."""

import argparse, math, os, sys, time, glob, csv, warnings
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import resize as tv_resize
from pytorch_msssim import ms_ssim
from models import CLC, GRCL

warnings.filterwarnings("ignore")

# ─── Metrics ──────────────────────────────────────────────────────────────────
def psnr(a, b):  return -10 * math.log10(torch.mean((a - b)**2).item())
def msssim_db(a, b):  return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())

def pad(x, p=128):
    h, w = x.size(2), x.size(3)
    nh, nw = (h+p-1)//p*p, (w+p-1)//p*p
    pl, pt = (nw-w)//2, (nh-h)//2
    return F.pad(x, (pl, nw-w-pl, pt, nh-h-pt), value=0), (pl, nw-w-pl, pt, nh-h-pt)

def unpad(x, pads):
    return F.pad(x, (-pads[0], -pads[1], -pads[2], -pads[3]))

def bd_rate(r1, psnr1, r2, psnr2):
    """Bjøntegaard delta rate (%) — simplified log-domain polynomial fit."""
    from scipy.interpolate import interp1d
    lr1 = np.log(np.array(r1)); lr2 = np.log(np.array(r2))
    p1 = np.array(psnr1); p2 = np.array(psnr2)
    lo = max(p1.min(), p2.min()); hi = min(p1.max(), p2.max())
    if lo >= hi: return 0.0
    f1 = interp1d(p1, lr1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(p2, lr2, kind='linear', fill_value='extrapolate')
    pts = np.linspace(lo, hi, 100)
    return (np.trapz(f2(pts) - f1(pts), pts) / (hi - lo)) * 100

# ─── Dataset (directory of images) ───────────────────────────────────────────
class ImageDirDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ref_dir=None, n_refs=3):
        self.files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                             if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
        self.ref_dir = ref_dir; self.n_refs = n_refs; self.to_t = transforms.ToTensor()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        x = self.to_t(img)
        refs = []
        if self.ref_dir:
            base = os.path.splitext(os.path.basename(self.files[idx]))[0]
            for k in range(self.n_refs):
                rp = os.path.join(self.ref_dir, f"{base}_ref{k}.png")
                refs.append(self.to_t(Image.open(rp).convert('RGB')) if os.path.exists(rp)
                            else torch.zeros_like(x))
        return x, refs, os.path.basename(self.files[idx])

# ─── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(net, dataset, device, use_ref=True, p=128):
    net.eval()
    results = []
    for x, refs, name in dataset:
        x = x.unsqueeze(0).to(device)
        if use_ref and refs:
            refs = [tv_resize(r.unsqueeze(0), list(x.shape[-2:])).to(device) for r in refs]
        else:
            refs = None
        xp, pads = pad(x, p)
        rp = [pad(r, p)[0] for r in refs] if refs else None

        t0 = time.time()
        enc = net.compress(xp, rp)
        dec = net.decompress(enc['strings'], enc['shape'], rp)
        dt = time.time() - t0

        x_hat = unpad(dec['x_hat'], pads)
        npx = x.size(2) * x.size(3)
        bpp = sum(len(s[0]) for s in enc['strings']) * 8.0 / npx
        p_val = psnr(x, x_hat)
        results.append({'name': name, 'bpp': bpp, 'psnr': p_val, 'time': dt})
        print(f"  {name}: {bpp:.4f} bpp | {p_val:.2f} dB | {dt:.2f}s")
    avg_bpp = np.mean([r['bpp'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])
    print(f"  AVG: {avg_bpp:.4f} bpp | {avg_psnr:.2f} dB")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="grcl", choices=["clc", "grcl"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="directory of test images")
    p.add_argument("--ref_dir", default="", help="directory of reference images")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_refs", type=int, default=3)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.model == 'grcl':
        net = GRCL(N=args.N, M=320, num_slices=5, num_ref=args.n_refs, clm_type='simple')
    else:
        net = CLC(N=args.N, M=320, num_slices=5, max_support_slices=5, num_ref_frames=args.n_refs)
    net = net.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    net.load_state_dict(sd); net.update()

    ds = ImageDirDataset(args.data, args.ref_dir if args.ref_dir else None, args.n_refs)
    evaluate(net, ds, device, use_ref=bool(args.ref_dir))

if __name__ == "__main__":
    main()
