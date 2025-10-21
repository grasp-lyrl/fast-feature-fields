"""
A minimal F3 model implementation with PyTorch as the only dependency.
This is intended to be used with torch.hub.
Or you can hack the code here to suit your needs!

Author: Richeek Das
"""
dependencies = ['torch', 'yaml']

import yaml
import logging
from copy import copy
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

# Set up logger to print to stdout
logger = logging.getLogger("f3")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

MODEL_NAME_TO_CKPT = {
    "1280x720x20_patchff_ds1_small": "https://richeek-penn.s3.amazonaws.com/prolev/ff/patchff_fullcardaym3ed_small_20ms.pth"
    #! TODO: Add the rest of the models here
}

HERE = str(Path(__file__).parent.resolve())

def calculate_receptive_field(layers):
    """
    Calculate the receptive field for a sequence of layers.

    Parameters:
        layers: List of tuples [(kernel_size1, stride1, dilation1), (kernel_size2, stride2, dilation2), ...]

    Returns:
        Receptive field size at the final layer.
    """
    receptive_field = 1
    product = 1
    for kernel_size, stride, dilation in layers:
        product *= stride
        receptive_field += (kernel_size - 1) * dilation * product
    return receptive_field


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

        if drop_path > 0:
            raise NotImplementedError("Drop path not implemented for F3 on torch.hub")
        self.drop_path = nn.Identity()

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


class MultiResolutionHashEncoder(nn.Module):
    """
        I write separate forward functions for 2D and 3D because I can't put if statements inside the forward
        pass, otherwise torch.compile will throw all sorts of errors.
    """
    def __init__(self,
                 compile: bool = False,
                 **kwargs):
        super().__init__()

        self.PI1: int = 1
        self.PI2: int = 2654435761
        self.PI3: int = 805459861

        self.compile = compile
        self.D = kwargs.get("D", 3)
        self.L_H = kwargs.get("L_H", 2)
        self.L_NH = kwargs.get("L_NH", 6)
        self.polarity = kwargs.get("polarity", False)
        self.feature_size = kwargs.get("feature_size", 4)
        self.resolutions = kwargs.get("resolutions", None)
        self.levels = kwargs.get("levels", self.L_H + self.L_NH) # Should be equal to L_H + L_NH
        self.log2_entries_per_level = kwargs.get("log2_entries_per_level", 19)

        try:
            self.index = getattr(self, f"index{self.D}d")
            self.forward = getattr(self, f"forward{'_pol' if self.polarity else '_nopol'}")
        except AttributeError:
            raise ValueError(f"Invalid number of dimensions: {self.D}")
        logger.info(f"Using {self.D}D Multi-Resolution Hash Encoder with {self.L_H} hashed levels and {self.L_NH} non-hashed levels and polarity: {self.polarity}")

        def get_hashmap():
            with torch.no_grad():
                hashmap = torch.zeros((self.levels, 1 << self.log2_entries_per_level, self.feature_size), dtype=torch.float32)
                hashmap.uniform_(-1e-4, 1e-4)
                hashmap = nn.Parameter(hashmap) # L x T x F where T = 2^log2_entries_per_level, F = feature_size
            return hashmap

        # build the hash tables
        if not self.polarity:
            self.hashmap = get_hashmap()
        else:
            self.hashmap_neg = get_hashmap()
            self.hashmap_pos = get_hashmap()

    def hash_linear_congruential3d(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, log2_entries_per_level: int) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2 ^ t * self.PI3) % (1 << log2_entries_per_level)
    
    def hash_linear_congruential2d(self, x: torch.Tensor, y: torch.Tensor, log2_entries_per_level: int) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2) % (1 << log2_entries_per_level)
    
    def index3d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y, t = eventBlock[:, :, 0], eventBlock[:, :, 1], eventBlock[:, :, 2]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape == t.shape, f"x.shape: {x.shape}, y.shape: {y.shape}, t.shape: {t.shape}"
            assert x.min() >= 0 and x.max() <= 1, "x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, "y coordinate should be in [0, 1]"
            assert t.min() >= 0, "t should be non-negative"

        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0] # B x N x L
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1] # B x N x L
        scaled_t = t.unsqueeze(-1) * self.resolutions[:, 2] # B x N x L

        floor_scaled_x = scaled_x.int() # B x N x L
        floor_scaled_y = scaled_y.int() # B x N x L
        floor_scaled_t = scaled_t.int() # B x N x L

        ceil_scaled_x = torch.min(floor_scaled_x + 1, self.resolutions[:, 0][None, None, :]) # B x N x L
        ceil_scaled_y = torch.min(floor_scaled_y + 1, self.resolutions[:, 1][None, None, :]) # B x N x L
        ceil_scaled_t = torch.min(floor_scaled_t + 1, self.resolutions[:, 2][None, None, :]) # B x N x L

        # all combinations of the 8 corners of the cube B X N x L x 8 x 3
        corners = torch.stack([
            torch.stack([floor_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1)
        ], dim=-2).int()  # B x N x L x 8 x 3

        # calculate the weights for each corner B x N x L x 8
        weights = torch.prod(1 - (corners - torch.stack([scaled_x, scaled_y, scaled_t], dim=-1).unsqueeze(-2)).abs(), dim=-1)

        # Calculate the indices for the hash table (Hash + Non-hash levels depending on the resolution)
        # B x N x L_H x 8 where L_H is the number of levels that need hashing
        hash_values = self.hash_linear_congruential3d(corners[:,:,-self.L_H:,:,0],
                                                      corners[:,:,-self.L_H:,:,1],
                                                      corners[:,:,-self.L_H:,:,2],
                                                      self.log2_entries_per_level)
        # B x N x L_NH x 8 where L_NH is the number of levels that don't need hashing
        nonhash_values = corners[:,:,:self.L_NH,:,0] +\
                         corners[:,:,:self.L_NH,:,1] * self.resolutions[:self.L_NH,0][None,None,:,None] +\
                         corners[:,:,:self.L_NH,:,2] * (self.resolutions[:self.L_NH,0] * self.resolutions[:self.L_NH,1])[None,None,:,None]
        
        return hash_values, nonhash_values, weights
    
    def index2d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y = eventBlock[:, :, 0], eventBlock[:, :, 1]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
            assert x.min() >= 0 and x.max() <= 1, "x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, "y coordinate should be in [0, 1]"
        
        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0]
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1]

        floor_scaled_x = scaled_x.int()
        floor_scaled_y = scaled_y.int()

        ceil_scaled_x = torch.min(floor_scaled_x + 1, self.resolutions[:, 0][None, None, :])
        ceil_scaled_y = torch.min(floor_scaled_y + 1, self.resolutions[:, 1][None, None, :])

        corners = torch.stack([
            torch.stack([floor_scaled_x, floor_scaled_y], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y], dim=-1)
        ], dim=-2).int()

        weights = torch.prod(1 - (corners - torch.stack([scaled_x, scaled_y], dim=-1).unsqueeze(-2)).abs(), dim=-1)

        hash_values = self.hash_linear_congruential2d(corners[:,:,-self.L_H:,:,0],
                                                      corners[:,:,-self.L_H:,:,1],
                                                      self.log2_entries_per_level)
        nonhash_values = corners[:,:,:self.L_NH,:,0] +\
                         corners[:,:,:self.L_NH,:,1] * self.resolutions[:self.L_NH,0][None,None,:,None]

        return hash_values, nonhash_values, weights

    def forward_pol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the Multi-Resolution Hash Encoder for 4D events (i.e. polarities).


            Args:
                x: x coordinate of the event rescaled to [0, 1]. (B,N)
                y: y coordinate of the event rescaled to [0, 1]. (B,N)
                t: time bin of the event. (B,N)
                (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])
                p: polarity of the event. (B,N)

                (x,y,t,p) or (x,y,p)

            Returns:
                The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        p = eventBlock[:, :, -1].reshape(-1)
        B, N = eventBlock.shape[0], eventBlock.shape[1]
        hash_values, nonhash_values, weights = self.index(eventBlock)
        
        hash_values, nonhash_values = hash_values.reshape(B * N, self.L_H, 2**self.D), nonhash_values.reshape(B * N, self.L_NH, 2**self.D)
        neg_idx, pos_idx = (p == 0).nonzero().squeeze(), (p == 1).nonzero().squeeze() #! Boolean indexing doesn't seem to work with torch.compile
        hashmap_features = torch.zeros((B * N, self.levels, 2**self.D, self.feature_size), dtype=self.hashmap_neg.dtype, device=eventBlock.device)
        for i in range(self.L_NH):
            hashmap_features[neg_idx, i, :, :] = self.hashmap_neg[i][nonhash_values[neg_idx, i, :]]
            hashmap_features[pos_idx, i, :, :] = self.hashmap_pos[i][nonhash_values[pos_idx, i, :]]
        for i in range(self.L_H):
            hashmap_features[neg_idx, i + self.L_NH, :, :] = self.hashmap_neg[i + self.L_NH][hash_values[neg_idx, i, :]]
            hashmap_features[pos_idx, i + self.L_NH, :, :] = self.hashmap_pos[i + self.L_NH][hash_values[pos_idx, i, :]]
        hashmap_features = hashmap_features.reshape(B, N, self.levels, 2**self.D, self.feature_size)

        interpolated_features = torch.sum(weights.unsqueeze(-1) * hashmap_features, dim=-2)
        interpolated_features = interpolated_features.reshape(B, N, -1)
        return interpolated_features

    def forward_nopol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the Multi-Resolution Hash Encoder.

            Args:
                x: x coordinate of the event rescaled to [0, 1]. (B,N)
                y: y coordinate of the event rescaled to [0, 1]. (B,N)
                t: time bin of the event. (B,N)
                (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])

                (x,y,t) or (x,y)

            Returns:
                The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        B, N = eventBlock.shape[0], eventBlock.shape[1] # Batch size, Number of events in each batch
        hash_values, nonhash_values, weights = self.index(eventBlock)
        
        # We have a hashmap of size L x T x F where T = 2^log2_entries_per_level, F = feature_size
        # We want to index it with N x L x 8 index tensor. That is, each row in the 1st dim of the index matrix,
        # index into the rows of the hashmap tensor. The 8 corners index into the 1st dim of the hashmap tensor.
        # We don't want to use gather with expand, because in backward pass it goes OOM. Well I don't like loops,
        # but I couldn't find a cleverer way to use gather which doesn't either expand on the N or F dimension.
        hashmap_features = torch.zeros((B, N, self.levels, 2**self.D, self.feature_size), dtype=self.hashmap.dtype, device=eventBlock.device)
        for i in range(self.L_NH):
            hashmap_features[:, :, i, :, :] = self.hashmap[i][nonhash_values[:, :, i, :]]
        for i in range(self.L_H):
            hashmap_features[:, :, i + self.L_NH, :, :] = self.hashmap[i + self.L_NH][hash_values[:, :, i, :]] 

        interpolated_features = torch.sum(weights.unsqueeze(-1) * hashmap_features, dim=-2) # B x N x L x F
        interpolated_features = interpolated_features.reshape(B, N, -1) # B x N x (L*F)
        return interpolated_features


class EventPatchFF(nn.Module):
    """ ConvNet for Patch prediction
        
    Args:
        in_chans (int): Number of input image channels. Default: 8
        T (int): Number of frames to predict. Default: 20
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        decoder_embed_dim (int): Decoder embedding dimension. Default: 256
        patch_size (int): Patch size, for what each feature pixel predicts in the future. Default: 8
    """
    def __init__(self,
                 multi_hash_encoder: dict, # Args for the multi_hash_encoder
                 T = 20, # Number of frames to predict
                 frame_sizes: list[int]=[1280, 720, 20], # Input size
                 dims: list[int]=[96, 128], # Feature dimensions at each stage
                 convkernels: list[int]=[7, 5], # Kernel size at each stage
                 convdepths: list[int]=[3, 3], # Number of blocks at each stage
                 convbtlncks: list[int]=[4, 4], # Bottleneck factor
                 convdilations: list[int]=[1, 1], # Dilation factor
                 dskernels: list[int]=[5, 5], # Downsample kernel size
                 dsstrides: list[int]=[2, 2], # Downsample stride
                 patch_size=16,
                 use_decoder_block: bool=True,
                 return_logits: bool=False, # Whether to return the logits or not
                 return_feat: bool=False, # Whether to return the key intermediate features or not
                 variable_mode: bool=True, # Whether to use variable mode or not
                 single_batch_mode: bool=False, # Whether to use single batch mode or not. Important for torch/onnx/trt compilation
                 device: str="cuda",
                ):
        super().__init__()
        assert len(dims) == len(convkernels) == len(convdepths) == len(dskernels) == len(dsstrides),\
               "Length of dims, convkernels, and convdepths should be the same."
        self._nstages = len(dims) 

        self.device = device
        self.frame_sizes = frame_sizes
        self.w, self.h, self.t = frame_sizes

        self.T = T
        self.dims = dims
        self.convkernels = convkernels
        self.convdepths = convdepths
        self.convbtlncks = convbtlncks
        self.convdilations = convdilations
        self.dskernels = dskernels
        self.dsstrides = dsstrides
        self.patch_size = patch_size
        self.multi_hash_encoder_args = copy(multi_hash_encoder)
        self.feature_size = dims[-1]

        self.use_decoder_block = use_decoder_block
        self.variable_mode = variable_mode
        self.return_logits = return_logits
        self.return_feat = return_feat

        #! Single batch mode is for inference with variable mode where we process one batch of events at a time
        #! This is to enable full graph compilation with torch.compile and AOTInductor. Variable sized events
        #! with variable number of event batches is hard to support with full graph compilation. So for now, this
        #! is the way.
        self.single_batch_mode = single_batch_mode

        self.downsample = torch.prod(torch.tensor(dsstrides, device='cpu')).item()
        self.padding = (self.patch_size - 1) // 2

        assert self.downsample == self.patch_size, "Downsample should be equal to the patch size."
        
        # Move the multi_hash_encoder args to the device
        multi_hash_encoder["finest_resolution"] = torch.tensor(
            multi_hash_encoder["finest_resolution"], dtype=torch.float32, device=device
        )
        multi_hash_encoder["coarsest_resolution"] = torch.tensor(
            multi_hash_encoder["coarsest_resolution"], dtype=torch.float32, device=device
        )

        gp_resolution = (
            torch.log(multi_hash_encoder["finest_resolution"]) -\
            torch.log(multi_hash_encoder["coarsest_resolution"])
        ) / (multi_hash_encoder["levels"] - 1)

        resolutions = torch.exp(
            torch.log(multi_hash_encoder["coarsest_resolution"]).unsqueeze(0) +\
            torch.arange(multi_hash_encoder["levels"], device=device).unsqueeze(1) * gp_resolution.unsqueeze(0)
        ).type(torch.int32) # since data minibatches are always on GPU

        multi_hash_encoder["resolutions"] = resolutions
        multi_hash_encoder["L_H"] = (torch.log2(resolutions[:,:3] + 1).sum(dim=1) >\
                                     multi_hash_encoder["log2_entries_per_level"]).sum().item()
        multi_hash_encoder["L_NH"] = multi_hash_encoder["levels"] - multi_hash_encoder["L_H"]

        # Multi-resolution hash encoder for the feature field generating events.
        logger.info("Instantiating Multi-resolution Hash Encoder with the following resolutions: "+\
                         f"{torch.Tensor(frame_sizes[:3]) / multi_hash_encoder['resolutions'][:, :3].cpu()}")
        self.multi_hash_encoder = MultiResolutionHashEncoder(compile=True, **multi_hash_encoder)

        in_channels = multi_hash_encoder["feature_size"] * multi_hash_encoder["levels"]

        kernel_strides_dilations = []
        for i in range(self._nstages):
            kernel_strides_dilations.append((dskernels[i], dsstrides[i], 1))
            kernel_strides_dilations.extend([(convkernels[i], 1, convdilations[i]) for _ in range(convdepths[i])])
        spatial_receptive_field = calculate_receptive_field(kernel_strides_dilations)
        logger.info(f"Spatial receptive field: {spatial_receptive_field}")

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=dskernels[0], stride=dsstrides[0], padding=(dskernels[0]-1)//2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        for i in range(1, self._nstages):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i-1], dims[i], kernel_size=dskernels[i], stride=dsstrides[i], padding=(dskernels[i]-1)//2),
            ))

        self.stages = nn.ModuleList()
        for i in range(self._nstages):
            self.stages.append(nn.Sequential(*[
                Block(dim=dims[i], kernel_size=convkernels[i], bottleneck=convbtlncks[i], dilation=convdilations[i])
                for _ in range(convdepths[i])
            ]))

        if self.use_decoder_block:
            self.decoder = Block(dim=dims[-1], kernel_size=7)
        self.pred = nn.Conv2d(dims[-1], patch_size**2 * T, kernel_size=1)

        self.apply(self._init_weights)
        
        """
        1280, 720, 8   -> 4x4 Conv changing channels and downsample by 2  + 3       = 3
        640, 360, 64   -> 7x7 block with 64 channels                      + 6 * 2   = 15
        640, 360, 64   -> 7x7 block with 64 channels                      + 6 * 2   = 27
        640, 360, 64   -> 4x4 Conv stride 2                               + 3 * 2   = 33
        320, 180, 128  -> 7x7 block with 128 channels                     + 6 * 4   = 57
        320, 180, 128  -> 7x7 block with 128 channels                     + 6 * 4   = 81
        320, 180, 128  -> 4x4 Conv stride 4                               + 3 * 4   = 93
        80, 45, 256    -> 7x7 block with 256 channels                     + 4 * 16  = 157
        80, 45, 256    -> 1x1 Conv 512
        
        80, 45, 512    -> 1x1 Conv 512 -> 16x16xT
        80, 45, 16x16xT
        
        OR
        
        1280, 720, 8   -> 5x5 Conv changing channels and downsample by 2  + 4       = 4
        640, 360, 64   -> 7x7 block with 64 channels                      + 6 * 2   = 16
        640, 360, 64   -> 7x7 block with 64 channels                      + 6 * 2   = 28
        640, 360, 64   -> 7x7 block with 64 channels                      + 6 * 2   = 40
        640, 360, 64   -> 5x5 Conv stride 2                               + 4 * 2   = 48
        320, 180, 128  -> 5x5 block with 128 channels                     + 4 * 4   = 64
        320, 180, 128  -> 5x5 block with 128 channels                     + 4 * 4   = 80
        320, 180, 128  -> 5x5 block with 128 channels                     + 4 * 4   = 96
        
        320, 180, 128  -> 1x1 Conv 128 -> 16x16xT
        """
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def unpatchify(self, x):
        """
            x: (N, patch_size**2 * T, W/downsample, H/downsample)
            imgs: (N, W*downsample, H*downsample, T)
        """
        if self.patch_size == 1:
            return x.permute(0, 2, 3, 1)
        else:
            p = self.patch_size
            n, _, w, h = x.shape
            x = x.permute(0, 2, 3, 1).reshape(n, w, h, p, p, -1)
            imgs = x.permute(0, 1 ,3, 2, 4, 5).reshape(n, w * p, h * p, -1)
            return imgs

    @classmethod
    def init_from_config(cls, eventff_config: str, return_logits: bool=False,
                         return_feat: bool=False, **kwargs):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            multi_hash_encoder=conf["multi_hash_encoder"], T = conf["T"], frame_sizes=conf["frame_sizes"],
            dims=conf["dims"], convkernels=conf["convkernels"], convdepths=conf["convdepths"],
            convbtlncks=conf["convbtlncks"], convdilations=conf["convdilations"],
            dskernels=conf["dskernels"], dsstrides=conf["dsstrides"],
            patch_size=conf["patch_size"], use_decoder_block=conf["use_decoder_block"],
            return_logits=return_logits, return_feat=return_feat,
            device=conf.get("device", "cuda"), variable_mode=conf.get("variable_mode", True),
            **kwargs
        )

    def _build_hashed_feats(self):
        # Precompute the hashed features for all possible event locations
        w, h, t = self.w, self.h, self.t
        xs = torch.arange(w, device=self.device) / w
        ys = torch.arange(h, device=self.device) / h
        ts = torch.arange(t, device=self.device) / t
        stacked = torch.stack(torch.meshgrid(xs, ys, ts, indexing='ij'), dim=-1).reshape(-1, 3)  # (w*h*t, 3)
        self.hashed_feats = torch.zeros(w, h, t, self.multi_hash_encoder.feature_size * self.multi_hash_encoder.levels, device=self.device)
        with torch.no_grad():
            # chunk to avoid OOM
            chunk_size = 100000
            for i in range(0, stacked.shape[0], chunk_size):
                end = min(i + chunk_size, stacked.shape[0])
                hashed_chunk = self.multi_hash_encoder(stacked[i:end][None, ...])[0]  # (chunk_size, L*F)
                self.hashed_feats.view(-1, self.multi_hash_encoder.feature_size * self.multi_hash_encoder.levels)[i:end] = hashed_chunk

    def save_config(self, path: str):
        with open(path, "w") as f:
            yaml.dump({
                "model": "EventPatchFF", "multi_hash_encoder": self.multi_hash_encoder_args, "T": self.T,
                "frame_sizes": self.frame_sizes, "dims": self.dims, "convkernels": self.convkernels,
                "convdepths": self.convdepths, "convbtlncks": self.convbtlncks, "convdilations": self.convdilations,
                "dskernels": self.dskernels, "dsstrides": self.dsstrides, "patch_size": self.patch_size,
                "use_decoder_block": self.use_decoder_block, "variable_mode": self.variable_mode
            }, f, default_flow_style=None)

    def forward_fixed(self, currentBlock: torch.Tensor) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B,N,3/4)
        """
        B, N = currentBlock.shape[0], currentBlock.shape[1]
        curr_x = (currentBlock[:,:,0] * self.w).round().int()
        curr_y = (currentBlock[:,:,1] * self.h).round().int()

        encoded_events = self.multi_hash_encoder(currentBlock) # (B,N,3)/(B,N,4) -> (B,N,L*F)

        feature_field = torch.zeros((B, self.w, self.h, encoded_events.shape[-1]), device=encoded_events.device)
        batch_indices = torch.arange(B).to(encoded_events.device).view(-1, 1).expand(B, N)
        feature_field.index_put_((batch_indices, curr_x, curr_y), encoded_events, accumulate=True)
        return feature_field

    def forward_variable(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,)    [N1, N2, N3, ..., NB]
        """
        B = eventCounts.shape[0]
        curr_x = (currentBlock[:,0] * self.w).round().int()
        curr_y = (currentBlock[:,1] * self.h).round().int()

        encoded_events = self.multi_hash_encoder(currentBlock.unsqueeze(0)).clone().squeeze(0) # (N,3)/(N,4) -> (N,L*F)

        feature_field = torch.zeros((B, self.w, self.h, encoded_events.shape[-1]), device=encoded_events.device)
        batch_indices = torch.repeat_interleave(eventCounts).int() # Offending line, cant compile. I am not spending time on this for now.
        feature_field.index_put_((batch_indices, curr_x, curr_y), encoded_events, accumulate=True)
        return feature_field.permute(0, 3, 1, 2)  # (B, C, W, H)

    def forward_variable_single(self, currentBlock: torch.Tensor) -> torch.Tensor:
        """
        This is for single batch variable mode inference. This is important for torch/onnx/trt compilation
        because it enables full graph compilation. And in actual deployment scenarios, we usually process
        one batch of events at a time anyway. Variable number of events per batch is still supported.

            currentBlock: torch.Tensor (N, 3/4) N = N1 + N2 + N3 + ... + NB
        """
        curr_x = (currentBlock[:,0] * self.w).round().int().clamp(0, self.w - 1)
        curr_y = (currentBlock[:,1] * self.h).round().int().clamp(0, self.h - 1)
        # AOTI export doesnt work without the clamping

        if getattr(self, "hashed_feats", None) is None:
            encoded_events = self.multi_hash_encoder(currentBlock[None, ...])[0] # (N,3)/(N,4) -> (N,L*F)
        else: # this is slightly faster at the cost of a lot more memory
            curr_t = (currentBlock[:,2] * self.t).round().long().clamp(0, self.t - 1)
            encoded_events = self.hashed_feats[curr_x, curr_y, curr_t]  # shape: (N, LF)

        feature_field = torch.zeros((self.w, self.h, encoded_events.shape[-1]),
                                    device=encoded_events.device,
                                    dtype=encoded_events.dtype)
        feature_field.index_put_((curr_x, curr_y), encoded_events, accumulate=True)
        return feature_field.permute(2, 0, 1)[None, ...]  # (1, C, W, H)

    def forward(self, currentBlock: torch.Tensor, eventCounts: Optional[torch.Tensor]=None) -> Optional[tuple[torch.Tensor, ...]]:
        """
            currentBlock: torch.Tensor (B, N, 3/4) or (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,) [N1, N2, N3, ..., NB]
        """
        if not self.return_logits and not self.return_feat:
            return None

        if self.variable_mode:
            if self.single_batch_mode:
                x = self.forward_variable_single(currentBlock)  # (1, C, W, H)
            else:
                x = self.forward_variable(currentBlock, eventCounts) # (B,C,W,H)
        else:
            if self.single_batch_mode:
                raise ValueError("Inference mode with fixed mode is not supported.")
            else:
                x = self.forward_fixed(currentBlock) # (B,C,W,H)

        for i in range(self._nstages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if self.return_feat:
            feat = x.permute(0, 2, 3, 1)

        if  not self.return_logits and self.return_feat:
            return feat

        if self.use_decoder_block:
            x = self.decoder(x)
        logits = self.pred(x) # (B,C,W,H) -> (B,patch_size**2*T,W/ds,H/ds)

        logits = self.unpatchify(logits) # (B,patch_size**2*T,W/ds,H/ds) -> (B,W,H,T)

        ret = (logits,)
        if self.return_feat: ret += (feat,)
        return ret


def compile_and_freeze_params(_model, compile=True, freeze=False):
    if compile:
        _model = torch.compile(
            _model,
            fullgraph=False,
            backend="inductor",
            options={
                "epilogue_fusion": True,
                "max_autotune": True,
            },
        )
    if freeze:
        for param in _model.parameters():
            param.requires_grad = False
        _model.eval()
    return _model

def load_f3_weights_ckpt(obj, eventff_ckpt: str, strict: bool=True):
    if eventff_ckpt.startswith(('http://', 'https://')):
        ckpt_dict = torch.hub.load_state_dict_from_url(eventff_ckpt, weights_only=True,
                                                       progress=True)
    else:
        ckpt_dict = torch.load(eventff_ckpt, weights_only=True)
    obj.load_state_dict(ckpt_dict["model"], strict=strict)
    epoch = ckpt_dict["epoch"]
    loss = ckpt_dict["loss"]
    acc = ckpt_dict["acc"]
    del ckpt_dict
    torch.cuda.empty_cache()
    return epoch, loss, acc

def init_f3_model(eventff_config: str, return_feat: bool=False, **kwargs) -> nn.Module:
    with open(eventff_config, "r") as f:
        conf = yaml.safe_load(f)
        model = conf["model"]
    if model == "EventPatchFF":
        model = EventPatchFF
    else:
        raise ValueError(f"Unknown model: {model}")
    return model.init_from_config(eventff_config, return_feat=return_feat, **kwargs)


def f3(pretrained: bool=False, compile: bool=True, **kwargs) -> nn.Module:
    """Load the Fast Feature Field (F3) model via PyTorch Hub

    This is a PyTorch Hub entrypoint for loading F3 models with various configurations.

        pretrained (bool, optional): If True, returns a model with pre-trained weights.
            Defaults to False.
        name (str): Required. The name of the configuration file (without .yml extension)
            that specifies the model architecture and parameters.
        **kwargs: Additional arguments to pass to the model initialization.

    Returns:
        nn.Module: The initialized F3 model, optionally with pre-trained weights.

    Raises:
        AssertionError: If 'name' argument is not provided or if the specified 
            configuration file doesn't exist.
        AssertionError: If pretrained=True but the model name is not found in 
            available pre-trained checkpoints.

    Example usage:
        # Load model with randomly initialized weights
        model = torch.hub.load('grasp-lyrl/fast-feature-fields', 'f3',
                              name='model_config')

        # Load model with pre-trained weights
        model = torch.hub.load('repo_owner/repo_name', 'f3', 
                              name='model_config', pretrained=True)

        # Load model with additional parameters
        model = torch.hub.load('repo_owner/repo_name', 'f3', 
                              name='model_config', custom_param=value)

    Note: 'name' is a required argument and must correspond to a valid configuration file 
    located in the 'confs/ff/modeloptions/' directory of the repository. Example config:

    model = torch.hub.load('grasp-lyrl/fast-feature-fields', 'f3',
                          name='1280x720x20_patchff_ds1_small', pretrained=True,
                          return_feat=True, return_logits=False)
    """
    assert "name" in kwargs, "Please provide the config file name with the 'name' argument."

    name = kwargs.pop("name")
    cfg = Path(HERE) / 'confs/ff/modeloptions' / f'{name}.yml'
    assert Path(cfg).is_file(), f"Config file {cfg} does not exist."

    model = init_f3_model(cfg, **kwargs)
    model = compile_and_freeze_params(model, compile=compile, freeze=False)

    if pretrained:
        assert name in MODEL_NAME_TO_CKPT, f"Model name {name} not found in MODEL_NAME_TO_CKPT"
        epoch, loss, acc = load_f3_weights_ckpt(model, MODEL_NAME_TO_CKPT[name])
        logger.info(f"Loaded pretrained model from epoch {epoch} with loss {loss} and acc {acc}")

    return model


if __name__ == "__main__":
    def setup_torch(cudnn_benchmark: bool=False):
        torch.manual_seed(403)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.benchmark = cudnn_benchmark # turn on for faster training if we are using the fixed event mode
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.compiled_autograd = True

    setup_torch(cudnn_benchmark=False)

    cfg = "confs/ff/modeloptions/1280x720x20_patchff_ds1_small.yml"
    ckpt = "outputs/patchff_fullcardaym3ed_small_20ms/models/best.pth"

    f3 = compile_and_freeze_params(init_f3_model(cfg, return_feat=True)).cuda()
    epoch, loss, acc = load_f3_weights_ckpt(f3, ckpt)

    logger.info(f"Loaded model from epoch {epoch} with loss {loss} and acc {acc}")
    torch.cuda.empty_cache()
