"""
GRCL — Generative Reference of Conditional Latents  (TPAMI 2025)

Architecture: TCM backbone (CNN + Swin-Transformer) with
  - CLM (Conditional Latent Matching)  — similarity matching + deformable alignment
  - CLS (Conditional Latent Synthesis)  — adaptive α-blending in latent space
  - Hyperprior + slice-based autoregressive entropy model conditioned on ref features

Key design choices following the paper:
  1. Encoder/decoder share g_a for input and references (deterministic at both sides).
  2. CLM+CLS operate only at the encoder to produce y_f; the decoder reconstructs y_f
     from the bitstream and only needs compact ref_features for entropy conditioning.
  3. ref_features are derived from g_a(ref).detach() so that reference encoding does
     NOT back-propagate into g_a to avoid training instability (Sec. IV, Eq. 8-9).
"""

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock, ResidualBlock, ResidualBlockUpsample,
    ResidualBlockWithStride, conv3x3, subpel_conv3x3,
)

import torch, torch.nn as nn, torch.nn.functional as F, math, numpy as np
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from .CLM import CLM, SimpleCLM
from .CLS import CLS

# ── helpers ──────────────────────────────────────────────────────────────────
SCALES_MIN, SCALES_MAX, SCALES_LEVELS = 0.11, 256, 64

def conv1x1(c_in, c_out, s=1):
    return nn.Conv2d(c_in, c_out, 1, s)

def conv_k(c_in, c_out, k=5, s=2):
    return nn.Conv2d(c_in, c_out, k, s, k // 2)

def get_scale_table():
    return torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def _update_registered_buffer(mod, name, key, sd, policy="resize_if_empty", dtype=torch.int):
    buf = next((b for n, b in mod.named_buffers() if n == name), None)
    if buf is None: raise RuntimeError(f'buffer "{name}" not registered')
    if policy == "resize" or buf.numel() == 0: buf.resize_(sd[key].size())

def update_registered_buffers(mod, prefix, names, sd):
    if not mod: return
    for n in names:
        _update_registered_buffer(mod, n, f"{prefix}.{n}", sd)


# ── Swin-Transformer building blocks (from TCM) ─────────────────────────────

class WMSA(nn.Module):
    def __init__(self, dim, out_dim, head_dim, ws, tp):
        super().__init__()
        self.dim, self.head_dim, self.ws, self.tp = dim, head_dim, ws, tp
        self.scale = head_dim ** -0.5; self.nh = dim // head_dim
        self.embed = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, out_dim)
        rpp = nn.Parameter(torch.zeros((2*ws-1)*(2*ws-1), self.nh)); trunc_normal_(rpp, std=.02)
        self.rpp = nn.Parameter(rpp.view(2*ws-1, 2*ws-1, self.nh).transpose(1,2).transpose(0,1))

    def _mask(self, h, w, p, s):
        m = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.rpp.device)
        if self.tp == 'W': return m
        t = p - s
        m[-1,:,:t,:,t:,:]=True; m[-1,:,t:,:,:t,:]=True
        m[:,-1,:,:t,:,t:]=True; m[:,-1,:,t:,:,:t]=True
        return rearrange(m, 'a b c d e f -> 1 1 (a b) (c d) (e f)')

    def _rel(self):
        c = torch.tensor([[i,j] for i in range(self.ws) for j in range(self.ws)])
        r = c[:,None,:] - c[None,:,:] + self.ws - 1
        return self.rpp[:, r[:,:,0].long(), r[:,:,1].long()]

    def forward(self, x):
        ws = self.ws
        if self.tp != 'W': x = torch.roll(x, (-(ws//2), -(ws//2)), (1,2))
        x = rearrange(x, 'b (h p1) (w p2) c -> b h w p1 p2 c', p1=ws, p2=ws)
        H, W = x.size(1), x.size(2)
        x = rearrange(x, 'b h w p1 p2 c -> b (h w) (p1 p2) c', p1=ws, p2=ws)
        q, k, v = rearrange(self.embed(x), 'b n s (t c)->t b n s c', c=self.head_dim).chunk(3, 0)
        s = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale + rearrange(self._rel(), 'h p q->h 1 1 p q')
        if self.tp != 'W': s = s.masked_fill_(self._mask(H, W, ws, ws//2), float('-inf'))
        o = torch.einsum('hbwij,hbwjc->hbwic', F.softmax(s, -1), v)
        o = self.proj(rearrange(o, 'h b w p c -> b w p (h c)'))
        o = rearrange(o, 'b (h w) (p1 p2) c -> b (h p1) (w p2) c', h=H, p1=ws)
        if self.tp != 'W': o = torch.roll(o, (ws//2, ws//2), (1,2))
        return o


class SwinBlock(nn.Module):
    def __init__(self, dim, hd, ws, dp):
        super().__init__()
        self.b1 = _Block(dim, hd, ws, dp, 'W')
        self.b2 = _Block(dim, hd, ws, dp, 'SW')
        self.ws = ws
    def forward(self, x):
        if x.size(-1) <= self.ws or x.size(-2) <= self.ws:
            pr = (self.ws - x.size(-2))//2; pc = (self.ws - x.size(-1))//2
            x = F.pad(x, (pc, pc+1, pr, pr+1))
        t = Rearrange('b c h w -> b h w c')(x)
        t = self.b2(self.b1(t))
        return Rearrange('b h w c -> b c h w')(t)

class _Block(nn.Module):
    def __init__(self, dim, hd, ws, dp, tp):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.msa = WMSA(dim, dim, hd, ws, tp)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        x = x + self.dp(self.msa(self.ln1(x)))
        return x + self.dp(self.mlp(self.ln2(x)))


class ConvTransBlock(nn.Module):
    def __init__(self, cd, td, hd, ws, dp, tp='W'):
        super().__init__()
        self.cd, self.td = cd, td
        self.tb = _Block(td, hd, ws, dp, tp)
        self.c1 = nn.Conv2d(cd+td, cd+td, 1); self.c2 = nn.Conv2d(cd+td, cd+td, 1)
        self.cb = ResidualBlock(cd, cd)
    def forward(self, x):
        cx, tx = torch.split(self.c1(x), (self.cd, self.td), 1)
        cx = self.cb(cx) + cx
        tx = Rearrange('b c h w->b h w c')(tx); tx = self.tb(tx); tx = Rearrange('b h w c->b c h w')(tx)
        return x + self.c2(torch.cat((cx, tx), 1))


class SWAtten(AttentionBlock):
    def __init__(self, i_d, o_d, hd, ws, dp, inter=192):
        n = inter or i_d; super().__init__(N=n)
        self.non_local_block = SwinBlock(n, hd, ws, dp)
        if inter: self.in_conv = conv1x1(i_d, n); self.out_conv = conv1x1(n, o_d)
    def forward(self, x):
        x = self.in_conv(x); ident = x; z = self.non_local_block(x)
        return self.out_conv(self.conv_a(x) * torch.sigmoid(self.conv_b(z)) + ident)


# ── GRCL Model ───────────────────────────────────────────────────────────────

class GRCL(CompressionModel):
    """Generative Reference of Conditional Latents — compression model."""

    def __init__(self, config=[2]*6, head_dim=[8,16,32,32,16,8],
                 drop_path_rate=0, N=128, M=320, num_slices=5,
                 max_support_slices=5, num_ref=3, clm_type='simple', **kw):
        super().__init__(entropy_bottleneck_channels=N)
        self.num_slices = num_slices; self.max_ss = max_support_slices
        self.num_ref = num_ref; self.M = M; self.N = N; self.ws = 8
        S = M // num_slices  # slice channels
        ref_dim = 64
        dim = N; dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        def _ctb(cfg_i, hd):
            return [ConvTransBlock(dim, dim, hd, self.ws, dpr[i], 'W' if not i%2 else 'SW') for i in range(config[cfg_i])]

        # ── g_a / g_s ────────────────────────────────
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, 2*N, 2), *_ctb(0, head_dim[0]), ResidualBlockWithStride(2*N, 2*N, 2),
            *_ctb(1, head_dim[1]), ResidualBlockWithStride(2*N, 2*N, 2),
            *_ctb(2, head_dim[2]), conv3x3(2*N, M, stride=2))
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, 2*N, 2), *_ctb(3, head_dim[3]), ResidualBlockUpsample(2*N, 2*N, 2),
            *_ctb(4, head_dim[4]), ResidualBlockUpsample(2*N, 2*N, 2),
            *_ctb(5, head_dim[5]), subpel_conv3x3(2*N, 3, 2))

        # ── CLM + CLS ────────────────────────────────
        self.clm = (CLM(M, min(M,256), 0.07, num_ref) if clm_type == 'full'
                     else SimpleCLM(M, 0.07, num_ref))
        self.cls = CLS(M, 128)

        # ── h_a / h_s (hyperprior) ────────────────────
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, 2*N, 2),
            *[ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') for i in range(config[0])],
            conv3x3(2*N, 192, stride=2))
        def _hs():
            return nn.Sequential(ResidualBlockUpsample(192, 2*N, 2),
                *[ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') for i in range(config[3])],
                subpel_conv3x3(2*N, M, 2))
        self.h_mean_s = _hs(); self.h_scale_s = _hs()

        # ── entropy context (ref-enhanced + baseline) ─
        def _cc(extra):
            return nn.ModuleList(nn.Sequential(
                conv_k(M+S*min(i,5)+extra, 224, 3, 1), nn.GELU(),
                conv_k(224, 128, 3, 1), nn.GELU(),
                conv_k(128, S, 3, 1)) for i in range(num_slices))
        def _lrp(extra):
            return nn.ModuleList(nn.Sequential(
                conv_k(M+S*min(i+1,6)+extra, 224, 3, 1), nn.GELU(),
                conv_k(224, 128, 3, 1), nn.GELU(),
                conv_k(128, S, 3, 1)) for i in range(num_slices))
        def _attn():
            return nn.ModuleList(nn.Sequential(
                SWAtten(M+S*min(i,5), M+S*min(i,5), 16, self.ws, 0, 128))
                for i in range(num_slices))

        self.att_mu = _attn(); self.att_sc = _attn()
        self.cc_mu = _cc(0); self.cc_sc = _cc(0)          # baseline (no ref)
        self.rcc_mu = _cc(ref_dim); self.rcc_sc = _cc(ref_dim)  # ref-enhanced
        self.lrp = _lrp(0); self.rlrp = _lrp(ref_dim)

        # ref feature adapter (projects concat ref latents → compact)
        self.ref_adapt = nn.Sequential(conv1x1(M * num_ref, 128), nn.GELU(), conv1x1(128, ref_dim))

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def update(self, scale_table=None, force=False):
        updated = self.gaussian_conditional.update_scale_table(scale_table or get_scale_table(), force)
        return updated | super().update(force=force)

    def _encode_refs(self, refs):
        """Encode references; .detach() so ref gradients don't leak into g_a."""
        if refs is None: return None, None
        lats = [self.g_a(r).detach() for r in refs]        # ← detach prevents data leakage
        feat = self.ref_adapt(torch.cat(lats, 1))
        return lats, feat

    def _clm_cls(self, y, ref_lats):
        if ref_lats is None: return y, None
        y_a = self.clm(y, ref_lats)
        y_f, info = self.cls(y, y_a)
        return y_f, info

    # ── slice-based autoregressive loop ────────────────────────────────────────

    def _slice_loop(self, y_f, lat_mu, lat_sc, ref_feat, y_shape, mode='forward'):
        """Shared logic for forward / compress / decompress slice iteration."""
        slices_in = y_f.chunk(self.num_slices, 1) if mode != 'decompress' else [None]*self.num_slices
        hat_slices, mu_l, sc_l, lh_l = [], [], [], []
        has_ref = ref_feat is not None

        # For decompress we need decoder state
        decoder = None
        if mode == 'decompress':
            decoder = self._dec_state

        for i in range(self.num_slices):
            sup = hat_slices if self.max_ss < 0 else hat_slices[:self.max_ss]
            m_sup = self.att_mu[i](torch.cat([lat_mu]+sup, 1))
            s_sup = self.att_sc[i](torch.cat([lat_sc]+sup, 1))

            if has_ref:
                mu = self.rcc_mu[i](torch.cat([m_sup, ref_feat], 1))
                sc = self.rcc_sc[i](torch.cat([s_sup, ref_feat], 1))
            else:
                mu = self.cc_mu[i](m_sup); sc = self.cc_sc[i](s_sup)
            mu = mu[:,:,:y_shape[0],:y_shape[1]]; sc = sc[:,:,:y_shape[0],:y_shape[1]]
            mu_l.append(mu); sc_l.append(sc)

            if mode == 'forward':
                _, lh = self.gaussian_conditional(slices_in[i], sc, mu)
                lh_l.append(lh); y_hat_s = ste_round(slices_in[i] - mu) + mu
            elif mode == 'compress':
                idx = self.gaussian_conditional.build_indexes(sc)
                yq = self.gaussian_conditional.quantize(slices_in[i], "symbols", mu)
                self._sym.extend(yq.reshape(-1).tolist()); self._idx.extend(idx.reshape(-1).tolist())
                y_hat_s = yq + mu
            else:  # decompress
                idx = self.gaussian_conditional.build_indexes(sc)
                rv = decoder.decode_stream(idx.reshape(-1).tolist(), *self._cdfs)
                rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
                y_hat_s = self.gaussian_conditional.dequantize(rv, mu)

            # LRP
            if has_ref:
                lrp = self.rlrp[i](torch.cat([m_sup, y_hat_s, ref_feat], 1))
            else:
                lrp = self.lrp[i](torch.cat([m_sup, y_hat_s], 1))
            y_hat_s = y_hat_s + 0.5 * torch.tanh(lrp)
            hat_slices.append(y_hat_s)

        return hat_slices, mu_l, sc_l, lh_l

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, ref_frames=None):
        y = self.g_a(x)
        ref_lats, ref_feat = self._encode_refs(ref_frames)
        y_f, cls_info = self._clm_cls(y, ref_lats)

        z = self.h_a(y_f); _, z_lh = self.entropy_bottleneck(z)
        z_hat = ste_round(z - self.entropy_bottleneck._get_medians()) + self.entropy_bottleneck._get_medians()
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)

        hat_slices, _, _, lh_l = self._slice_loop(y_f, lat_mu, lat_sc, ref_feat, y_f.shape[2:], 'forward')
        y_hat = torch.cat(hat_slices, 1)
        return {'x_hat': self.g_s(y_hat), 'likelihoods': {'y': torch.cat(lh_l,1), 'z': z_lh}}

    # ── compress / decompress ────────────────────────────────────────────────

    def compress(self, x, ref_frames=None):
        y = self.g_a(x)
        ref_lats, ref_feat = self._encode_refs(ref_frames)
        y_f, _ = self._clm_cls(y, ref_lats)

        z = self.h_a(y_f); z_str = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_str, z.size()[-2:])
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)

        self._sym, self._idx = [], []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_l = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        off = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        self._cdfs = (cdf, cdf_l, off)

        self._slice_loop(y_f, lat_mu, lat_sc, ref_feat, y_f.shape[2:], 'compress')
        enc = BufferedRansEncoder(); enc.encode_with_indexes(self._sym, self._idx, *self._cdfs)
        return {'strings': [[enc.flush()], z_str], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape, ref_frames=None):
        _, ref_feat = self._encode_refs(ref_frames)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)
        y_shape = [z_hat.shape[2]*4, z_hat.shape[3]*4]

        self._cdfs = (self.gaussian_conditional.quantized_cdf.tolist(),
                      self.gaussian_conditional.cdf_length.reshape(-1).int().tolist(),
                      self.gaussian_conditional.offset.reshape(-1).int().tolist())
        self._dec_state = RansDecoder(); self._dec_state.set_stream(strings[0][0])

        hat_slices, *_ = self._slice_loop(None, lat_mu, lat_sc, ref_feat, y_shape, 'decompress')
        return {'x_hat': self.g_s(torch.cat(hat_slices, 1)).clamp_(0, 1)}

    # ── state dict ───────────────────────────────────────────────────────────

    def load_state_dict(self, state_dict):
        update_registered_buffers(self.gaussian_conditional, "gaussian_conditional",
                                  ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        own = self.state_dict()
        super().load_state_dict({k: v for k, v in state_dict.items() if k in own}, strict=False)
