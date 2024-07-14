import mmsr.models.archs.arch_util as arch_util
from mmsr.models.archs.vgg_arch import VGGFeatureExtractor
from mmsr.models.archs.sync_batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from torch.nn import functional as F
import math
import einops
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as init
from einops import rearrange


try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
     
        out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)
    

class Mlp(nn.Module):
    def __init__(self, input_resolution, in_features, hidden_features=None, out_features=None, act_layer="gelu", drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = GeLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.input_resolution = input_resolution

        self.conv_layer = nn.Sequential(
                default_conv(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3),
                nn.GELU()
            )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W).contiguous()
        x = self.conv_layer(x)
        x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def unfold(input: torch.Tensor,
           window_size: int,
           kernel_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    channels, height, width = input.shape  # type: int, int, int
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=0, size=window_size, step=window_size) \
        .unfold(dimension=0, size=kernel_size, step=kernel_size)

    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(5, 0, 1, 4, 2, 3)
    B, H, W, H_, W_, C = output.shape
    output = output.reshape(-1, H * W, H_ * W_, C)

    return output


def fold(input: torch.Tensor,
         window_size: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    # H, W: int = int(input.shape[2] // (window_size), int(input.shape[3]// window_size)
    # Reshape input to
    output: torch.Tensor = input.view(-1, input.shape[2] // window_size, window_size, input.shape[3] // window_size, window_size, channels).contiguous()
    B, H, W, H_, W_, C = output.shape
    output: torch.Tensor = output.reshape(B*W*W_, H*H_, channels)
    return output


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')  


def gelu(x):
    """Implementation of the GeLU() activation function.
        For information: OpenAI GPT's GeLU() is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the GeLU() activation function.
        For information: OpenAI GPT's GeLU() is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

ACT2FN = {"gelu": gelu}

import torch.nn.utils.spectral_norm as spectral_norm
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        H, W = self.input_resolution

        self.norm = SynchronizedBatchNorm2d(in_channel, affine=False)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv_0 = spectral_norm(self.mlp_gamma)
        self.conv_1 = spectral_norm(self.mlp_beta)

        norm_layer = nn.LayerNorm
        self.normalized = norm_layer(in_channel, style_dim)

    def forward(self, input, actv):
        
        B, L, C = input.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"       

        norm = self.normalized(input)
        style = self.style(actv)
        new_style = torch.squeeze(style)
        gamma, beta = new_style.chunk(2, 1)
  
        # apply scale and bias
        out = norm * gamma + beta

        return out


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    
    

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):
        super().__init__()
        ### Standard attn block 
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.gating_param = nn.Parameter(torch.ones(self.num_heads), requires_grad= True)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
            
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.trunk = TrunkBranch(dim, kernel_size=3, stride=1, padding=1)

        self.proj_q = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, padding=0
        )

        self.softmax = nn.Softmax(dim=-1)

        # self.apply(self._init_weights)    

        
    def forward(self, x, ref, input_resolution, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        H, W = input_resolution
        B_, N, C = x.shape

        x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
        x = rearrange(x, 'B H W C -> B C H W').contiguous()
        x = self.qkv(x)
        x = rearrange(x, 'B C H W -> B H W C').contiguous()
        x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
        qkv = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        ###---------------------------------------------------------------------------------------
        ref = window_reverse(ref, self.window_size[0], H, W)  # (B, H, W, C)
        ref = rearrange(ref, 'B H W C -> B C H W').contiguous()
        ref = self.qkv(ref)
        ref = rearrange(ref, 'B C H W -> B H W C').contiguous()
        ref = window_partition(ref, self.window_size[0])  # num_windows*B, w, w, C
        qkv_ref = rearrange(ref, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_ref = qkv_ref.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_ref, k_ref, v_ref = qkv_ref[0], qkv_ref[1], qkv_ref[2]  # make torchscript happy (cannot use tensor as tuple)     
       
      
        attn1 = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) # [B*wh*ww, nheads, WH*WW, WH*WW]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn1 = attn1 * logit_scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn1 = attn1 + relative_position_bias.unsqueeze(0)

        attn2 = (F.normalize(q, dim=-1) @ F.normalize(k_ref, dim=-1).transpose(-2, -1)) # [B*wh*ww, nheads, WH*WW, WH*WW]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn2 = attn2 * logit_scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn2 = attn2 + relative_position_bias.unsqueeze(0)


        if mask is not None:
            nW = mask.shape[0]
            attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N)
            attn1 = self.softmax(attn1)

            attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N)
            attn2 = self.softmax(attn2)

        else:
            attn1 = self.softmax(attn1)
            attn2 = self.softmax(attn2)

        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        gating = (self.gating_param).view(1,-1,1,1) 
        attn = (1.-torch.sigmoid(gating)) * attn2 + torch.sigmoid(gating) * attn1

        gating = (self.gating_param).view(1,-1,1,1) 
        attn = (1.-torch.sigmoid(gating)) * attn1 + torch.sigmoid(gating) * attn2

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
        x = rearrange(x, 'B H W C -> B C H W').contiguous()
        x = self.trunk(x)
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C').contiguous()
        x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
        x = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C

        x = self.proj_drop(x)
        return x
        
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    

class ScaleNorm(nn.Module):
    """See
    https://github.com/lucidrains/reformer-pytorch/blob/a751fe2eb939dcdd81b736b2f67e745dc8472a09/reformer_pytorch/reformer_pytorch.py#L143
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class RefSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: GeLU()
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=GeLU(), use_mlp=True, norm_layer=nn.LayerNorm, fused_window_process=False, fc=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp = use_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.qkv = nn.Linear(dim, 3*dim, bias=True)
        self.qkv_conv = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=3, stride=1, padding=1, bias=qkv_bias)
        self.proj_ = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim*2, dim)
        self.proj_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
           
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)

        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

           # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                            self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        if self.use_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(input_resolution=input_resolution, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(input_resolution=input_resolution, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
   
        self.gamma1 = nn.Parameter(1e-4 * torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-4 * torch.ones((dim)), requires_grad=True)  
        self.linear = nn.Linear(dim * 2, dim)
        self.norm = nn.InstanceNorm1d(dim)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
     
   
    def forward(self, x, ref):
        H, W = self.input_resolution
       
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        ref_shortcut = ref
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        ref = ref.view(B, H, W, C)

        x_1 = x[:, :, :, :]
        x_3 = ref[:, :, :, :]
        if self.shift_size > 0:
            x_2 = torch.roll(x[:, :, :, :], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_4 = torch.roll(ref[:, :, :, :], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_2 = x[:, :, :, :]
            x_4 = ref[:, :, :, :]

        # partition windows
        x1_windows = window_partition(x_1, self.window_size)  # nW*B, window_size, window_size, C
        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x3_windows = window_partition(x_3, self.window_size)  # nW*B, window_size, window_size, C
        x3_windows = x3_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x2_windows = window_partition(x_2, self.window_size)  # nW*B, window_size, window_size, C
        x2_windows = x2_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x4_windows = window_partition(x_4, self.window_size)  # nW*B, window_size, window_size, C
        x4_windows = x4_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
         
        if self.input_resolution == self.input_resolution:
            attn_1 = self.attn(x1_windows, x3_windows, mask=self.attn_mask1, input_resolution=self.input_resolution)  # nW*B, window_size*window_size, C
            attn_2 = self.attn(x2_windows, x4_windows, mask=self.attn_mask2, input_resolution=self.input_resolution)  # nW*B, window_size*window_size, C
        else:
            attn_1 = self.attn(x1_windows, x3_windows, mask=self.calculate_mask(self.input_resolution).to(x.device), x_size=self.input_resolution)
            attn_2 = self.attn(x2_windows, x4_windows, mask=self.calculate_mask(self.input_resolution).to(x.device), x_size=self.input_resolution)

        # merge windows
        attn_1 = attn_1.view(-1, self.window_size, self.window_size, C)
        attn_2 = attn_2.view(-1, self.window_size, self.window_size, C)
        attn_1 = window_reverse(attn_1, self.window_size, H, W)  # B H W C
        attn_2 = window_reverse(attn_2, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_2 = torch.roll(attn_2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_2 = attn_2

        x = torch.cat([attn_1.reshape(B, H * W, C), attn_2.reshape(B, H * W, C)], dim=2)

        ref = ref.reshape(B, H * W, C)
        x = self.proj(x)

        # FFN
        x = shortcut + self.drop_path(x)
        ref = ref_shortcut + self.drop_path(ref)
        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm1(x)))
            ref = ref + self.drop_path(self.mlp(self.norm1(ref)))

        return x, ref
    

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
                
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows
    
    def get_window_ref(self, q):
        C = q.shape[-1]
                
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False, fused_window_process=False, fc=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RefSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size = window_size//2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=dim)
        else:
            self.upsample = None
        if all(v is not None for v in [downsample, upsample]):
            self.linear = SinusoidalPositionalEmbedding(embedding_dim=dim//2, padding_idx=0, init_size=dim // 2)
        

    def forward(self, x, ref):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, ref = checkpoint.checkpoint(blk, x, ref)
            else:
                x, ref = blk(x, ref)
        if self.downsample is not None:
            x = self.downsample(x)
            ref = self.downsample(ref)
        if self.upsample is not None:
            x = self.upsample(x)
            ref = self.upsample(ref)
        return x, ref

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 160.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 128.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=160, patch_size=4, in_chans=3, embed_dim=512, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
        

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, blur_kernel=[1, 3, 3, 1], out_dim=None, scale_factor=2):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(dim , dim*4, 3, 1, 1)
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=dim // 2, padding_idx=0, init_size=dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(self.conv(x))
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # # Add SPE    
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # LN
        flops = 4 * H * W * self.dim
        # proj
        flops += 4 * H * W * self.dim * (self.out_dim)
        # SPE
        flops += 4 * H * W * 2
        # bilinear
        flops += 4 * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GeLU() -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GeLU() -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = GeLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x 


class ResidualBlock_(nn.Module):
    """Residual block with BN.
    It has a style of:
        ---Conv-BN-ReLU-Conv-BN-+-
         |______________________|
    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 128.
        bn_affine (bool): Whether to use affine in BN layers. Default: True.
    """

    def __init__(self, nf=128, bn_affine=True):
        super(ResidualBlock_, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(nf, affine=True)
        self.conv2 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.bn2 = nn.BatchNorm2d(nf, affine=True)
        self.relu = nn.ReLU(inplace=True)

        default_init_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return identity + out

def default_init_weights(module_list, scale=1):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(basic_block, n_basic_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        n_basic_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(n_basic_blocks):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=128, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat
    
   
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 128.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        sn (bool): Whether to use spectral norm. Default: False.
        n_power_iterations (int): Used in spectral norm. Default: 1.
        sn_bias (bool): Whether to apply spectral norm to bias. Default: True.
    """

    def __init__(self,
                 nf=128,
                 res_scale=1,
                 pytorch_init=False,
                 sn=False,
                 n_power_iterations=1,
                 sn_bias=True):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sn:
            self.conv1 = spectral_norm(
                self.conv1,
                name='weight',
                n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(
                self.conv2,
                name='weight',
                n_power_iterations=n_power_iterations)
            if sn_bias:
                self.conv1 = spectral_norm(
                    self.conv1,
                    name='bias',
                    n_power_iterations=n_power_iterations)
                self.conv2 = spectral_norm(
                    self.conv2,
                    name='bias',
                    n_power_iterations=n_power_iterations)
        self.relu = nn.ReLU(inplace=True)

        if not sn and not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    
class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(
        self, n_feat, kernel_size, stride, padding,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    

## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
        self, n_feat, kernel_size, stride, padding,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(n_feat, kernel_size, stride, padding, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


    
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 160
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 128
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=40, embed_dim=48, depths=[2, 2], num_heads=[4, 4],
                 window_size=[8, 8, 8, 8, 4], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.ByteTensor = torch.cuda.ByteTensor

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        in_channels = [
            64, 
            64, 
            64, 
            64
            ] 
        
        self.layers_ref_s = nn.ModuleList()
        self.layers_ref_m = nn.ModuleList()
        self.layers_ref_g = nn.ModuleList()

                
        for i_layer in range(self.num_layers):
            in_channel = in_channels[1]
            layer_ref_m = BasicLayer(dim=in_channel,
                                input_resolution=(img_size*2, img_size*2),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size[i_layer],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint)
            self.layers_ref_m.append(layer_ref_m)

        for i_layer_ref in range(self.num_layers):
            in_channel = in_channels[0]
            layer_ref_s = BasicLayer(dim=in_channel,
                                        input_resolution=(img_size, img_size),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        use_checkpoint=use_checkpoint)
            self.layers_ref_s.append(layer_ref_s)

        for i_layer_ref in range(self.num_layers):
            in_channel = in_channels[2]
            layer_ref_g = BasicLayer(dim=in_channel,
                                        input_resolution=(img_size*4, img_size*4),
                                        depth=depths[i_layer_ref],
                                        num_heads=num_heads[i_layer_ref],
                                        window_size=window_size[i_layer_ref],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer_ref]):sum(depths[:i_layer_ref + 1])],
                                        norm_layer=norm_layer,
                                        use_checkpoint=use_checkpoint)
            self.layers_ref_g.append(layer_ref_g)

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 8, p2 = 8),
            nn.LayerNorm(in_channels[-1]),
            nn.Linear(in_channels[-1], in_channels[-1]),
            nn.LayerNorm(in_channels[-1]),
        )
        self.head = nn.Conv2d(6, in_channels[-1], kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels[-1], 3, 3, 1, 1)
        self.final_out = nn.Conv2d(9, 3, 3, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=in_channels[-1], n_blocks=16)
        self.ref_feat = CorrespondenceGenerationArch()
        self.conv = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(256, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.rg = RG(in_channels[-1], num_rcab=16, reduction=8)
        self.down = nn.Conv2d(in_channels[-1], in_channels[-1], 3, 2, 1)
        self.res = ResBlock(in_channels[-1], kernel_size=3, stride=1, padding=1)
        self.trunk = TrunkBranch(in_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_shuffle_up = nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-1]*4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2))
        self.last_shuffle_up = nn.Sequential(nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=3, stride=1, padding=1),
                                             nn.PixelShuffle(4))

        self.to_q_global_1 = nn.Sequential(
                FeatExtract(in_channels[-1], keep_dim=False),
                FeatExtract(in_channels[-1], keep_dim=False),
                FeatExtract(in_channels[-1], keep_dim=False),
            )
        self.to_q_global_2 = nn.Sequential(
                FeatExtract(in_channels[-1], keep_dim=False),
                FeatExtract(in_channels[-1], keep_dim=False),
            )
        self.to_q_global_3 = nn.Sequential(
                FeatExtract(in_channels[-1], keep_dim=False),
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def forward_features(self, lr, ref_g): 
        shortcut = lr
        edges = self.get_edges(lr)
        lr = self.to_q_global_1(self.head(torch.cat((lr, edges), 1)))
        in_features = self.ref_feat(ref_g)

        ref_s = self.to_q_global_1(self.conv1(in_features['relu3_1']))
        ref_so = ref_s
        ref_m = self.to_q_global_2(self.conv2(in_features['relu2_1']))
        ref_mo = ref_m
        ref_g = self.to_q_global_3(self.conv3(in_features['relu1_1']))
        ref_go = ref_g
                     
        base = F.interpolate(shortcut, None, 4, 'bilinear', False)
        ############ 
        lr1 = self.conv(torch.cat((lr, ref_so), 1))
        lr = self.trunk(lr1)

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_s = ref_so.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        for layer in self.layers_ref_s:
            lr, ref_s = layer(lr, ref_s)  

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2], shortcut.shape[3]).contiguous()  
        lr_111 = lr
        lr = self.mid_shuffle_up(lr)
        # ############ 
        lr2 = self.conv(torch.cat((lr, ref_mo), 1))
        lr = self.trunk(lr2)

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_m = ref_mo.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        for layer in self.layers_ref_m:
            lr, ref_m = layer(lr, ref_m)   

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*2, shortcut.shape[3]*2).contiguous()  
        lr_222 = lr
        ###################
        lr = self.down(lr)
        lr1 = self.conv(torch.cat((lr, ref_so), 1))
        lr = self.trunk(lr1)
        lr = self.trunk(self.conv(torch.cat((lr, lr_111), 1)))

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_s = ref_so.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        for layer in self.layers_ref_s:
            lr, ref_s = layer(lr, ref_s)  

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2], shortcut.shape[3]).contiguous()  
        lr_333 = lr
        lr = self.mid_shuffle_up(lr)
        #####################
        lr2 = self.conv(torch.cat((lr, ref_mo), 1))
        lr = self.trunk(lr2)
        lr = self.trunk(self.conv(torch.cat((lr, lr_222), 1)))

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_m = ref_mo.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        for layer in self.layers_ref_m:
            lr, ref_m = layer(lr, ref_m)   

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*2, shortcut.shape[3]*2).contiguous()  
        lr_444 = lr
        lr = self.trunk(self.conv(torch.cat((lr, lr_222), 1)))
        ###############
        lr = self.down(lr)
        lr = self.trunk(self.conv(torch.cat((lr, ref_so), 1)))
        lr = self.trunk(self.conv(torch.cat((lr, lr_111), 1)))
        lr = self.trunk(self.conv(torch.cat((lr, lr_333), 1)))

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_s = ref_so.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        for layer in self.layers_ref_s:
            lr, ref_s = layer(lr, ref_s)  

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2], shortcut.shape[3]).contiguous()  
        lr_555 = lr
        
        lr = self.mid_shuffle_up(lr)
        # ############ 
        lr2 = self.conv(torch.cat((lr, ref_mo), 1))
        lr = self.trunk(lr2)
        lr = self.trunk(self.conv(torch.cat((lr, lr_222), 1)))
        lr = self.trunk(self.conv(torch.cat((lr, lr_444), 1)))

        B, C, H, W = lr.shape
        lr = lr.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        ref_m = ref_mo.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        for layer in self.layers_ref_m:
            lr, ref_m = layer(lr, ref_m)   

        lr = lr.transpose(-1, -2).reshape(lr.shape[0], self.embed_dim, shortcut.shape[2]*2, shortcut.shape[3]*2).contiguous()  

        ################
        lr = self.mid_shuffle_up(lr)
        lr = self.trunk(lr)
        out = self.out(lr)
        return out + base 

    def forward(self, lr, ref_g):
        out = self.forward_features(lr, ref_g)

        return out             

    def flops(self):
        flops = 0
        flops += self.patch_embed_s.flops() 
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution_s[0] * self.patches_resolution_s[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class CorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19'):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def forward(self, img_ref_hr):
        batch_offset_relu3 = []
        batch_offset_relu2 = []
        batch_offset_relu1 = []

        img_ref_feat = self.vgg(img_ref_hr)
        return img_ref_feat
