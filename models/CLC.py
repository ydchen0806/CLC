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

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x

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
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
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
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
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
        self.similarity = nn.Conv2d(dim*2, 1, kernel_size=1)
        
    def forward(self, y, Y_r):
        y_m = self.conv(y)
        # S = self.similarity(torch.cat([y_m.unsqueeze(1).repeat(1, Y_r.size(1), 1, 1, 1), 
        #                                Y_r.unsqueeze(2).repeat(1, 1, y_m.size(2), 1, 1)], dim=2))
        # 将 Y_r 的第 1 维 (num_references) 和第 2 维 (channels) 合并
        Y_r_reshaped = Y_r.view(Y_r.size(0), -1, Y_r.size(3), Y_r.size(4))  # [batch_size, num_references * channels, height, width]

        # 拼接 y_m 和 Y_r_reshaped
        S = self.similarity(torch.cat([
            y_m.unsqueeze(1).repeat(1, Y_r.size(1), 1, 1, 1).view(y_m.size(0), -1, y_m.size(2), y_m.size(3)),  # 将 y_m 转换为与 Y_r_reshaped 兼容的 4D 张量
            Y_r_reshaped], dim=1))  # 在 channel 维度拼接
        S = F.softmax(S, dim=1)
        y_m = (S * Y_r).sum(dim=1)
        return y_m

class CLS(nn.Module):
    def __init__(self, dim):
        super(CLS, self).__init__()
        self.conv_mu = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_sigma = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        
    def forward(self, y, y_a):
        mu = self.conv_mu(torch.cat([y, y_a], dim=1))
        sigma = F.softplus(self.conv_sigma(torch.cat([y, y_a], dim=1)))
        y_f = mu + sigma * torch.randn_like(mu)
        return y_f, mu, sigma

class TCM(CompressionModel):
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
            ResidualBlockWithStride(3, 2*N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[0])],
            ResidualBlockWithStride(2*N, 2*N, stride=2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[1])],
            ResidualBlockWithStride(2*N, 2*N, stride=2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[2])],
            conv3x3(2*N, M, stride=2)
        )
        
        # Decoder
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M*2, 2*N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[3])],
            ResidualBlockUpsample(2*N, 2*N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[4])],
            ResidualBlockUpsample(2*N, 2*N, 2),
            *[ConvTransBlock(N, N, self.head_dim[i], self.window_size, 0, 'W' if not i%2 else 'SW') for i in range(config[5])],
            subpel_conv3x3(2*N, 3, 2)
        )
        
        # Hyperprior network
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, 2*N, 2),
            *[ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') for i in range(config[0])],
            conv3x3(2*N, 192, stride=2)
        )
        
        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(192, 2*N, 2),
            *[ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') for i in range(config[3])],
            subpel_conv3x3(2*N, M, 2)
        )
        
        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(192, 2*N, 2),
            *[ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') for i in range(config[3])],
            subpel_conv3x3(2*N, M, 2)
        )
        
        # CLM and CLS modules
        self.clm = CLM(M)
        self.cls = CLS(M)
        
        # CLM module for decoding
        self.clm_dec = CLM(M)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        
        # Context model
        self.context_prediction = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(M + M * i, M // 2, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(M // 2, M, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_slices)
        )

        # Entropy parameters prediction
        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(M * 2, M * 3 // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(M * 3 // 2, M * 2, 1),
            ) for _ in range(num_slices)
        )

    def forward(self, x, ref_x_list):
        # Encoding process
        y = self.g_a(x)
        
        # Process reference images
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        
        # CLM module
        y_m = self.clm(y, torch.stack(ref_y_list, dim=1))
        
        # CLS module
        y_f, mu, sigma = self.cls(y, y_m)
        
        # Hyperprior
        z = self.h_a(y_f)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y_f.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihoods = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu_slice = self.context_prediction[slice_index](mean_support)
            
            scale_support = torch.cat([latent_scales, mu_slice], dim=1)
            gaussian_params = self.entropy_parameters[slice_index](scale_support)
            scales_slice, means_slice = gaussian_params.chunk(2, 1)
            
            y_hat_slice = self.gaussian_conditional.quantize(
                y_slice, "noise" if self.training else "dequantize", means_slice
            )
            y_likelihoods.append(self.gaussian_conditional.likelihood(y_hat_slice, scales_slice, means_slice))
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": mu, "scales": sigma, "y": y_f}
        }

    def compress(self, x, ref_x_list):
        y = self.g_a(x)
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        
        y_m = self.clm(y, torch.stack(ref_y_list, dim=1))
        y_f, _, _ = self.cls(y, y_m)
        
        z = self.h_a(y_f)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y_f.chunk(self.num_slices, 1)
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_strings if self.max_support_slices < 0 else y_strings[:self.max_support_slices]
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu_slice = self.context_prediction[slice_index](mean_support)
            
            scale_support = torch.cat([latent_scales, mu_slice], dim=1)
            gaussian_params = self.entropy_parameters[slice_index](scale_support)
            scales_slice, means_slice = gaussian_params.chunk(2, 1)
            
            indexes = self.gaussian_conditional.build_indexes(scales_slice)
            y_string = self.gaussian_conditional.compress(y_slice, indexes, means_slice)
            y_strings.append(y_string)

            y_hat_slice = self.gaussian_conditional.quantize(y_slice, "dequantize", means_slice)
            latent_means = torch.cat([latent_means, y_hat_slice], dim=1)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, ref_x_list):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_hat = torch.zeros(
            (z_hat.size(0), self.M, z_hat.size(2) * 4, z_hat.size(3) * 4),
            device=z_hat.device,
        )

        for slice_index, y_string in enumerate(strings[0]):
            support_slices = (y_hat[:, :self.M * slice_index, :, :] 
                              if self.max_support_slices < 0 
                              else y_hat[:, :self.M * min(slice_index, self.max_support_slices), :, :])
            
            mean_support = torch.cat([latent_means, support_slices], dim=1)
            mu_slice = self.context_prediction[slice_index](mean_support)
            
            scale_support = torch.cat([latent_scales, mu_slice], dim=1)
            gaussian_params = self.entropy_parameters[slice_index](scale_support)
            scales_slice, means_slice = gaussian_params.chunk(2, 1)
            
            indexes = self.gaussian_conditional.build_indexes(scales_slice)
            y_hat_slice = self.gaussian_conditional.decompress(y_string, indexes, means=means_slice)
            
            y_hat[:, self.M * slice_index : self.M * (slice_index + 1), :, :] = y_hat_slice

        # Apply inverse CLS and CLM transforms
        ref_y_list = [self.g_a(ref_x) for ref_x in ref_x_list]
        y_m = self.clm_dec(y_hat, torch.stack(ref_y_list, dim=1))
        y, _, _ = self.cls(y_hat, y_m)
        
        x_hat = self.g_s(y).clamp_(0, 1)

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

# Test code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((1, 3, 256, 256)).to(device)
    ref_x_list = [torch.randn((1, 3, 256, 256)).to(device) for _ in range(3)]  # 3 reference images
    model = TCM().to(device)
    output = model(x, ref_x_list)
    print(output.keys())
    print(output['x_hat'].shape)
    print(output['likelihoods']['y'].shape)
    print(output['likelihoods']['z'].shape)