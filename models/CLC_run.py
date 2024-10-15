import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample, conv3x3, subpel_conv3x3
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
import torchvision.models as models
import math
from torch import Tensor
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def forward(self, x):
        if self.type != 'W': 
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': 
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]), device=self.relative_position_params.device)
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim  # e.g., 128
        self.trans_dim = trans_dim  # e.g., 128
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']

        # Convolutional path
        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)
        self.conv1_1 = nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # Transformer path
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_2 = nn.Conv2d(self.conv_dim, self.trans_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # Combine features from both paths
        self.conv1_3 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Convolutional path
        conv_x = self.conv1_1(x)  # [batch_size, conv_dim, H, W]
        conv_x = self.conv_block(conv_x) + conv_x  # Residual connection

        # Transformer path
        trans_x = self.conv1_2(x)  # [batch_size, trans_dim, H, W]
        trans_x = rearrange(trans_x, 'b c h w -> b h w c')
        trans_x = self.trans_block(trans_x)
        trans_x = rearrange(trans_x, 'b h w c -> b c h w')

        # Concatenate features and reduce channels
        combined = torch.cat((conv_x, trans_x), dim=1)  # [batch_size, conv_dim + trans_dim, H, W]
        res = self.conv1_3(combined)  # [batch_size, conv_dim, H, W]

        # Residual connection
        x = x + res  # Both x and res have conv_dim channels
        return x
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-3])
    
    def forward(self, x):
        return self.feature_extractor(x)

class CLM(nn.Module):
    def __init__(self, dim):
        super(CLM, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.similarity = nn.Conv2d(dim * 2, 1, kernel_size=1)
        
    def forward(self, y, Y_r):
        """
        y: [batch_size, channels, H, W]
        Y_r: [batch_size, num_references, channels, H, W]
        """
        y_m = self.conv(y)  # [batch_size, channels, H, W]
        batch_size, num_refs, channels, H, W = Y_r.size()
        Y_r = Y_r.view(batch_size * num_refs, channels, H, W)
        y_m_expanded = y_m.unsqueeze(1).repeat(1, num_refs, 1, 1, 1).view(batch_size * num_refs, channels, H, W)
        concat = torch.cat([y_m_expanded, Y_r], dim=1)  # [batch_size * num_refs, 2 * channels, H, W]
        S = self.similarity(concat)  # [batch_size * num_refs, 1, H, W]
        S = S.view(batch_size, num_refs, 1, H, W)
        S = F.softmax(S, dim=1)  # [batch_size, num_refs, 1, H, W]
        Y_r = Y_r.view(batch_size, num_refs, channels, H, W)
        y_m = (S * Y_r).sum(dim=1)  # [batch_size, channels, H, W]
        return y_m

class CLS(nn.Module):
    def __init__(self, dim):
        super(CLS, self).__init__()
        self.conv_mu = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.conv_sigma = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        
    def forward(self, y, y_a):
        mu = self.conv_mu(torch.cat([y, y_a], dim=1))
        sigma = F.softplus(self.conv_sigma(torch.cat([y, y_a], dim=1)))
        y_f = mu + sigma * torch.randn_like(mu)
        return y_f, mu, sigma

class CLC(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128, M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.M = M
        
        # Encoder
        self.g_a = nn.Sequential(
            conv3x3(3, N),
            ResidualBlockWithStride(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[0])],
            ResidualBlockWithStride(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[1])],
            ResidualBlockWithStride(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[2])],
            conv3x3(N, M, stride=2)
        )
        
        # Decoder
        self.g_s = nn.Sequential(
            conv3x3(M, N),
            ResidualBlockUpsample(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[3])],
            ResidualBlockUpsample(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[4])],
            ResidualBlockUpsample(N, N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if i % 2 == 0 else 'SW') for i in range(config[5])],
            subpel_conv3x3(N, 3, 2)
        )
        
        # Hyperprior network
        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
        )
        
        # **Adjusted Hyper synthesis network (h_s)**
        self.h_s = nn.Sequential(
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, M * 2),
        )
        
        # CLM and CLS modules
        self.clm = CLM(M)
        self.cls = CLS(M)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        # Context model
        self.context_prediction = nn.Conv2d(M, M * 2, 3, padding=1)
        
        # **Adjusted Entropy parameters prediction**
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 4, M * 2, 1),
            nn.GELU(),
            nn.Conv2d(M * 2, M * 2, 1),
            nn.GELU(),
            nn.Conv2d(M * 2, M * 2, 1),
        )
    
    def forward(self, x, ref_x_list):
        # Encoding process
        y = self.g_a(x)
        
        # Process reference images
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        
        # CLM module
        Y_r = torch.stack(ref_y_list, dim=1)  # [batch_size, num_refs, channels, H, W]
        y_m = self.clm(y, Y_r)
        
        # CLS module
        y_f, mu, sigma = self.cls(y, y_m)
        
        # Hyperprior
        z = self.h_a(y_f)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        params = self.h_s(z_hat)
        y_q = self.gaussian_conditional.quantize(y_f, 'noise' if self.training else 'dequantize', means=None)
        
        ctx_params = self.context_prediction(y_q)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales, means = gaussian_params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y_f, scales, means)
        
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": mu, "scales": sigma, "y": y_f}
        }
    
    def compress(self, x, ref_x_list):
        y = self.g_a(x)
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        
        Y_r = torch.stack(ref_y_list, dim=1)  # [batch_size, num_refs, channels, H, W]
        y_m = self.clm(y, Y_r)
        y_f, _, _ = self.cls(y, y_m)
        
        z = self.h_a(y_f)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.h_s(z_hat)
        
        y_shape = y_f.size()[-2:]
        y_strings = []
        for i in range(y_f.size(1)):
            y_i = y_f[:, i:i+1, :, :]
            y_q = ste_round(y_i)
            ctx_params = self.context_prediction(y_q)
            gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
            scales, means = gaussian_params.chunk(2, 1)
            indexes = self.gaussian_conditional.build_indexes(scales)
            y_string = self.gaussian_conditional.compress(y_i, indexes, means=means)
            y_strings.append(y_string)
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape, ref_x_list):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)
        
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        Y_r = torch.stack(ref_y_list, dim=1)  # [batch_size, num_refs, channels, H, W]
        
        y_hat = []
        for i, y_string in enumerate(strings[0]):
            if i == 0:
                y_i = torch.zeros((1, 1, shape[0] * 4, shape[1] * 4)).to(z_hat.device)
            else:
                y_i = torch.cat(y_hat, dim=1)
            ctx_params = self.context_prediction(y_i)
            gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
            scales, means = gaussian_params.chunk(2, 1)
            indexes = self.gaussian_conditional.build_indexes(scales)
            y_hat_i = self.gaussian_conditional.decompress(y_string, indexes, means=means)
            y_hat.append(y_hat_i)
        y_hat = torch.cat(y_hat, dim=1)
        
        y_m = self.clm(y_hat, Y_r)
        y_hat, _, _ = self.cls(y_hat, y_m)
        
        x_hat = self.g_s(y_hat)
        x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

# 使用 DDP 进行并行训练的代码示例
def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device('cuda', rank)
    model = CLC().to(device)
    # 使用 DDP 封装模型
    model = DDP(model, device_ids=[rank], output_device=rank)
    # 创建数据集和数据加载器
    # 这里需要使用 DistributedSampler
    # dataset = ...
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=..., sampler=sampler)
    # for data in dataloader:
    #     ...

    # 示例输入
    x = torch.randn((1, 3, 256, 256)).to(device)
    ref_x_list = [torch.randn((1, 3, 256, 256)).to(device) for _ in range(3)]  # 3 reference images

    output = model(x, ref_x_list)
    print(f"Rank {rank} output keys:", output.keys())
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    world_size = 1
    if world_size > 1:
        # 使用 torch.multiprocessing.spawn 启动多个进程
        torch.multiprocessing.spawn(main_worker, args=(world_size,), nprocs=world_size)
    else:
        # 单 GPU 训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn((1, 3, 256, 256)).to(device)
        ref_x_list = [torch.randn((1, 3, 256, 256)).to(device) for _ in range(3)]  # 3 reference images
        model = CLC().to(device)
        output = model(x, ref_x_list)
        print(output.keys())
        print(output['x_hat'].shape)
        print(output['likelihoods']['y'].shape)
        print(output['likelihoods']['z'].shape)