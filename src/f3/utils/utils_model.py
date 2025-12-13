import numpy as np
from types import NoneType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from timm.layers import DropPath

from .utils_gen import ev_to_frames, ev_to_grid

try:
    from torchsparse import SparseTensor    # type: ignore
    from torchsparse import nn as spnn      # type: ignore
    from torchsparse.nn.utils import fapply # type: ignore
except ImportError:
    SparseTensor = NoneType
    print("torchsparse not installed. Sparse operations will not work.")


def num_params(model: nn.Module)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_to_dict(model):
    model_dict = {"layers": []}
    for name, layer in model.named_children():
        layer_info = {
            "name": name,
            "type": type(layer).__name__,
            "parameters": {k: v.shape for k, v in layer.state_dict().items()}
        }
        model_dict["layers"].append(layer_info)
    return model_dict


class DotProd(nn.Module):
    def __init__(self):
        super(DotProd, self).__init__()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        # x: torch.Tensor of shape (B, N, 2*F) or (N, 2*F)
        half = x.size(-1) // 2 # assumes even
        half1 = x[..., :half] / torch.linalg.norm(x[..., :half], dim=-1, keepdim=True).clamp(min=1e-7)
        half2 = x[..., half:] / torch.linalg.norm(x[..., half:], dim=-1, keepdim=True).clamp(min=1e-7)
        return (((half1 * half2).sum(-1).unsqueeze(-1) + 1) / 2).clamp(0, 1) # (B, N, 1)


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, return_feat: bool = False, **kwargs):
        # Generates MLP with hidden dimensions [a_0, a_1, ..., a_N],
        # where a_0 is the input dimension and a_n is the output dimension
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        for cin, cout in zip(layer_sizes[:-1], layer_sizes[1:-1]):
            self.layers.append(nn.Linear(cin, cout, bias=bias))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=bias))
        self.weight_init()
        if return_feat: self.forward = self.forward_w_feat
        else:           self.forward = self.forward_wo_feat
    
    def weight_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward_wo_feat(self, x: torch.Tensor)->torch.Tensor:
        return self.layers(x)

    def forward_w_feat(self, x: torch.Tensor)->torch.Tensor:
        feat = self.layers[:-1](x)
        return self.layers[-1](feat), feat


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    def __init__(self, layer_sizes: list[int], w0: float = 30.0, normalize: bool = False):
        # Generates SIREN with hidden dimensions [a_0, a_1, ..., a_N],
        # where a_0 is the input dimension and a_n is the output dimension
        super(SIREN, self).__init__()
        self.w0 = w0
        self.layers = nn.Sequential()
        for cin, cout in zip(layer_sizes[:-1], layer_sizes[1:-1]):
            self.layers.append(nn.Linear(cin, cout))
            self.layers.append(Sine(w0=w0))
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if normalize: self.layers.append(nn.Sigmoid())
        self.weight_init()
    
    def weight_init(self):
        """
            Special initialization from the SIREN paper
        """
        with torch.no_grad():
            m = self.layers[0]
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)
            for m in self.layers[1:]:
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    m.weight.uniform_(-np.sqrt(6 / num_input) / self.w0, np.sqrt(6 / num_input) / self.w0)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.layers(x)


class SIREN_Scheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 end_factor: float = 0.1, last_epoch: int = -1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.end_factor = end_factor
        super(SIREN_Scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_lr = [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
            return warmup_lr
        else:
            # Linear decay
            decay_lr = [
                base_lr * ((self.total_steps - self.last_epoch) * (1 - self.end_factor) / (self.total_steps - self.warmup_steps) + self.end_factor)
                for base_lr in self.base_lrs
            ]
            return decay_lr


class PixelShuffleUpsample(nn.Module):
    """ PixelShuffle Upsampling layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int): Upscaling factor. Default: 2.
    
    Returns:
        torch.Tensor: Upsampled tensor.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2),
                              kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
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


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., kernel_size=7, bottleneck=4, dilation=1):
        super().__init__()
        if kernel_size > 1:
            padding = (kernel_size - 1) // 2 * dilation
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                    padding=padding, dilation=dilation, groups=dim) # depthwise conv
        else:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=1) # pointwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, bottleneck * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(bottleneck * dim)
        self.pwconv2 = nn.Linear(bottleneck * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class SparseLayerNorm(LayerNorm):
    """ SparseLayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps, "channels_last")
    
    def forward(self, x: SparseTensor) -> SparseTensor: # type: ignore
        return fapply(x, super().forward)


class SparseGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: SparseTensor) -> SparseTensor: # type: ignore
        Gx = torch.norm(x.feats, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        output = SparseTensor(
            coords=x.coords,
            feats=self.gamma * (x.feats * Nx) + self.beta + x.feats,
            stride=x.stride,
            spatial_range=x.spatial_range
        )
        output._caches = x._caches
        return output


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)
    
    def forward(self, x: SparseTensor) -> SparseTensor: # type: ignore
        return fapply(x, super().forward)
            

class SparseBlock(nn.Module):
    """ ConvNeXtV2 Sparse Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, kernel_size=7, bottleneck=4):
        super().__init__()
        if kernel_size > 1:
            self.dwconv = spnn.Conv3d(dim, dim, kernel_size=(kernel_size, kernel_size, 1), bias=True) # depthwise conv
        else:
            self.dwconv = spnn.Conv3d(dim, dim, kernel_size=1, bias=True) # pointwise conv
        self.norm = SparseLayerNorm(dim, eps=1e-6)
        self.pwconv1 = SparseLinear(dim, bottleneck * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = spnn.modules.ReLU()
        self.grn = SparseGRN(bottleneck * dim)
        self.pwconv2 = SparseLinear(bottleneck * dim, dim)

    def forward(self, x: SparseTensor) -> SparseTensor: # type: ignore
        input = x
        x = self.dwconv(x) # (N, F) -> (N', F)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + x
        return x


def to_dense(x: SparseTensor, b: int, c: int, w: int, h: int) -> torch.Tensor: # type: ignore
    coords = x.coords # (N, 4)
    feats = x.feats   # (N, F)
    dense = torch.zeros(b, c, w, h, dtype=feats.dtype).to(feats.device)
    dense[coords[:, 0], :, coords[:, 1], coords[:, 2]] = feats
    return dense


"""
Parts of the U-Net model 
Taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
class UNetDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDown(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNetUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = UNetDoubleConv(2 * in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


"""
EV2Frame and EV2VoxelGrid classes for converting event data to frames and voxel grids.
"""
class EV2Frame(nn.Module):
    def __init__(self, height: int=720, width: int=1280, polarity: bool=False):
        super(EV2Frame, self).__init__()
        self.height = height
        self.width = width
        self._chan = 1 + int(polarity)

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor) -> torch.Tensor:
        """
            eventBlock: (N, 3/4) tensor
            eventCounts: (B,) tensor

            Returns:
                frames: (B, 1, H, W) tensor
        """
        if self._chan == 2:
            pospolBlock = eventBlock[eventBlock[:, -1] == 1][:, :-1] # (N, 3)
            negpolBlock = eventBlock[eventBlock[:, -1] == 0][:, :-1] # (N, 3)

            cumsum = torch.cumsum(torch.cat((torch.zeros(1).to(eventCounts.device), eventCounts)), 0).to(torch.int32)
            pospolCounts, negpolCounts = eventCounts.clone(), eventCounts.clone()
            for i in range(eventCounts.size(0)):
                pospolCounts[i] = (eventBlock[cumsum[i]:cumsum[i+1], -1] == 1).sum()
                negpolCounts[i] = (eventBlock[cumsum[i]:cumsum[i+1], -1] == 0).sum()

            posframes = ev_to_frames(pospolBlock, pospolCounts, self.width, self.height) # (B, W, H) and 255
            negframes = ev_to_frames(negpolBlock, negpolCounts, self.width, self.height) # (B, W, H) and 255
            frames = torch.stack((posframes, negframes), dim=1).permute(0, 1, 3, 2) / 255.0 # (B, 2, H, W)
        else:
            frames = ev_to_frames(eventBlock, eventCounts, self.width, self.height) # (B, W, H) and 255
            frames = frames.unsqueeze(1).permute(0, 1, 3, 2) / 255.0 # (B, 1, H, W)
        return frames


class EV2VoxelGrid(nn.Module):
    def __init__(self, height: int=720, width: int=1280, timebins: int=20):
        super(EV2VoxelGrid, self).__init__()
        self.height = height
        self.width = width
        self.timebins = timebins
        self._chan = timebins

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor) -> torch.Tensor:
        """
            eventBlock: (B, N, 4) or (B, N, 3) tensor
            eventCounts: (B,) tensor
            
            Returns:
                voxelgrid: (B, T, H, W) tensor
        """
        voxelgrid = ev_to_grid(eventBlock, eventCounts, self.width, self.height, self.timebins) # (B, W, H, T) and 1
        voxelgrid = voxelgrid.permute(0, 3, 2, 1) # (B, T, H, W)
        return voxelgrid
