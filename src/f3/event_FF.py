import yaml
import logging
from copy import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from f3.utils import (MLP, SIREN, MultiResolutionHashEncoder, DotProd,
                          LayerNorm, Block, SparseLayerNorm, SparseBlock, EV2VoxelGrid,
                          UNetDoubleConv, UNetDown, UNetUp) #, CudaTimer
from f3.utils import num_params, LOSSES, to_dense, calculate_receptive_field

try:
    from torchsparse import nn as spnn   # type: ignore
    from torchsparse import SparseTensor # type: ignore
except ImportError:
    print("Please install torchsparse, if you want to use SparsePatchFF.")

from timm.layers import trunc_normal_


def get_mlp(mlp_type: str, dims: list[int]=None, **kwargs) -> nn.Module:
    """
        Get the MLP model based on the type.

        Args:
            mlp_type: str, the type of the MLP model.
            dims: list[int], the dimensions of the MLP model.

        Returns:
            The MLP model.
    """
    printer = kwargs.get("logger")
    printer = printer.info if printer else print
    if mlp_type == "mlp":
        printer(f"MLP size: {dims}")
        return MLP(dims, **kwargs)
    elif mlp_type == "siren":
        printer(f"SIREN MLP size: {dims}")
        return SIREN(dims, **kwargs)
    elif mlp_type == "dotprod":
        printer(f"Using Dot Product for prediction.")
        return DotProd()
    elif mlp_type == "sigmoid":
        printer(f"Using Sigmoid for prediction.")
        return nn.Sigmoid()
    else:
        raise ValueError(f"MLP type {mlp_type} not recognized.")


class EventPixelFF(nn.Module):
    """
        Event Feature Field. Deepset, Convolution, MLP Based.

        This network takes in the current block of events and predicts the next squashed frame of a specified time length.
        
        Args:
            multi_hash_encoder: dict, args for the multi-resolution hash encoder for the feature field generating events.
            frame_sizes: list[int], the frame size from which the current block of events is from.
            mlp_type: str, the type of the MLP model taking in the hash encoded and pooled event features.
            MLP_DIMS: list[int], the MLP hidden dimensions for the event predictor
            past_pool_size: int, the size of the past pool spatially.
    """
    def __init__(self,
                 multi_hash_encoder: dict, # Args for the multi_hash_encoder
                 T: int=20, # Number of frames to predict
                 frame_sizes: list[int]=[1280, 720, 50], # The frame size
                 MLP_DIMS: list[int]=[256, 256, 128, 64, 32], # The MLP hidden dimensions for the event predictor
                 past_pool_size: int=9,
                 return_feat: bool=False, # Whether to return the key intermediate features or not
                 return_loss: bool=False, # Whether to return the loss or not
                 loss_fn: str="VoxelFocalLoss", # Loss function to use
                 variable_mode: bool=True, # Whether to use variable mode or not
                 device: str="cuda",
                ):
        super().__init__()
        self.logger = logging.getLogger("__main__")
        assert past_pool_size % 2 == 1, f"Past pool size should be odd."

        self.w, self.h = frame_sizes[:2]
        self.patches = self.w * self.h
        self.past_pool_size = past_pool_size
        
        self.T = T
        self.MLP_DIMS = MLP_DIMS
        self.frame_sizes = frame_sizes
        self.multi_hash_encoder_args = copy(multi_hash_encoder)

        self.variable_mode = variable_mode
        self.return_feat = return_feat
        self.return_loss = return_loss
        if return_loss:
            self.loss_fn = LOSSES[loss_fn]        
        
        gp_resolution = (
            torch.log(torch.tensor(multi_hash_encoder["finest_resolution"], dtype=torch.float32)) -\
            torch.log(torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32))
        ) / (multi_hash_encoder["levels"] - 1)
        resolutions = torch.exp(
            torch.log(
                torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32)
            ).unsqueeze(0) + torch.arange(multi_hash_encoder["levels"]).unsqueeze(1) * gp_resolution.unsqueeze(0)
        ).type(torch.int32).to(device) # since data minibatches are always on GPU
        multi_hash_encoder["resolutions"] = resolutions
        multi_hash_encoder["L_H"] = (torch.log2(resolutions[:,:3] + 1).sum(dim=1) >\
                                     multi_hash_encoder["log2_entries_per_level"]).sum().item()
        multi_hash_encoder["L_NH"] = multi_hash_encoder["levels"] - multi_hash_encoder["L_H"]

        # Multi-resolution hash encoder for the feature field generating events.
        self.logger.info("Instantiating Multi-resolution Hash Encoder with the following resolutions: "+\
                         f"{torch.Tensor(frame_sizes[:3]) / multi_hash_encoder['resolutions'][:, :3].cpu()}")
        self.multi_hash_encoder = MultiResolutionHashEncoder(compile=compile, **multi_hash_encoder)

        in_channels = multi_hash_encoder["feature_size"] * multi_hash_encoder["levels"]
        self.convolution1 = nn.Conv2d(in_channels, 2*MLP_DIMS[0], kernel_size=past_pool_size,
                                      stride=1, padding=(past_pool_size-1)//2)
        self.convolution2 = nn.Conv2d(2*MLP_DIMS[0], 2*MLP_DIMS[0], kernel_size=past_pool_size,
                                      stride=1, padding=(past_pool_size-1)//2)
        self.convolution3 = nn.Conv2d(2*MLP_DIMS[0], 2*MLP_DIMS[0], kernel_size=past_pool_size,
                                      stride=1, padding=(past_pool_size-1)//2)
        self.convolution4 = nn.Conv2d(2*MLP_DIMS[0], MLP_DIMS[0], kernel_size=past_pool_size,
                                      stride=1, padding=(past_pool_size-1)//2)
        self.relu = nn.ReLU()
        self.event_predictor = get_mlp("mlp", MLP_DIMS + [T], bias=True, return_feat=return_feat, logger=self.logger)
    
    @classmethod
    def init_from_config(cls, eventff_config: str, return_feat: bool=False,
                         return_loss: bool=False, loss_fn: str="VoxelFocalLoss", **kwargs):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            multi_hash_encoder=conf["multi_hash_encoder"], T = conf["T"], frame_sizes=conf["frame_sizes"],
            MLP_DIMS=conf["MLP_DIMS"], past_pool_size=conf["patch_sizes"][0], return_loss=return_loss,
            return_feat=return_feat, loss_fn=loss_fn, device=conf.get("device", "cuda"),
            variable_mode=conf.get("variable_mode", True),
        )
    
    def save_config(self, path: str):
        with open(path, "w") as f:
            yaml.dump({
                "model": "EventPixelFF", "multi_hash_encoder": self.multi_hash_encoder_args, "T": self.T,
                "frame_sizes": self.frame_sizes, "MLP_DIMS": self.MLP_DIMS, "patch_sizes": [self.past_pool_size, 1],
                "variable_mode": self.variable_mode,
            }, f, default_flow_style=None)

    def forward_fixed(self, currentBlock: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B,N,3/4)
        """
        B, N = currentBlock.shape[0], currentBlock.shape[1]
        curr_x = (currentBlock[:,:,0] * w).round().int()
        curr_y = (currentBlock[:,:,1] * h).round().int()

        encoded_events = self.multi_hash_encoder(currentBlock) # (B,N,3)/(B,N,4) -> (B,N,L*F)

        feature_field = torch.zeros(B, encoded_events.shape[-1], w, h).to(encoded_events.device)
        batch_indices = torch.arange(B).to(encoded_events.device).view(-1, 1).expand(B, N)
        feature_field[batch_indices, :, curr_x, curr_y] += encoded_events

        return feature_field

    def forward_variable(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,)    [N1, N2, N3, ..., NB]
        """
        B, N = eventCounts.shape[0], currentBlock.shape[0]
        curr_x = (currentBlock[:,0] * self.w).round().int()
        curr_y = (currentBlock[:,1] * self.h).round().int()

        encoded_events = self.multi_hash_encoder(currentBlock.unsqueeze(0)).clone().squeeze(0) # (N,3)/(N,4) -> (N,L*F)

        feature_field = torch.zeros(B, encoded_events.shape[-1], self.w, self.h).to(encoded_events.device)
        batch_indices = torch.repeat_interleave(eventCounts) # Offending line, cant compile. I am not spending time on this for now.
        feature_field[batch_indices, :, curr_x, curr_y] += encoded_events
        return feature_field

    def forward_loss(self, predlogits: torch.Tensor, futureBlock: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
            predlogits: torch.Tensor (B, W, H, T)
            futureBlock: torch.Tensor (B, W, H, T)
        """
        return self.loss_fn(predlogits, futureBlock, valid_mask)

    def forward(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                futureBlock: torch.Tensor=None, valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B, N, 3/4) or (N, 3/4)
            eventCounts: torch.Tensor (B,)
            futureBlock: torch.Tensor (B, W, H, T)
        """
        if self.variable_mode:
            feature_field = self.forward_variable(currentBlock, eventCounts)
        else:
            feature_field = self.forward_fixed(currentBlock)
        
        feature_field = self.convolution1(feature_field) # (B,C,W,H) -> (B,C',W,H)
        feature_field = self.relu(feature_field)
        feature_field = self.convolution2(feature_field) # (B,C',W,H) -> (B,C',W,H)
        feature_field = self.relu(feature_field)
        feature_field = self.convolution3(feature_field)
        feature_field = self.relu(feature_field)
        feature_field = self.convolution4(feature_field)
        feature_field = self.relu(feature_field)
        feature_field = feature_field.permute(0, 2, 3, 1)

        if self.return_feat:
            predlogits, feature_field = self.event_predictor(feature_field)
        else:
            predlogits = self.event_predictor(feature_field)

        if self.return_loss:
            loss = self.forward_loss(predlogits, futureBlock, valid_mask)

        ret = (predlogits.squeeze(-1),)
        if self.return_feat: ret += (feature_field,)
        if self.return_loss: ret += (loss,)
        return ret        


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
                 return_loss: bool=False, # Whether to return the loss or not
                 loss_fn: str="VoxelFocalLoss", # Loss function to use
                 variable_mode: bool=True, # Whether to use variable mode or not
                 device: str="cuda",
                ):
        super().__init__()
        assert len(dims) == len(convkernels) == len(convdepths) == len(dskernels) == len(dsstrides),\
               f"Length of dims, convkernels, and convdepths should be the same."
        self._nstages = len(dims) 

        self.logger = logging.getLogger("__main__")
        self.frame_sizes = frame_sizes
        self.w, self.h = frame_sizes[:2]

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
        self.return_loss = return_loss
        if return_loss:
            self.loss_fn = LOSSES[loss_fn]

        self.downsample = torch.prod(torch.tensor(dsstrides)).item()
        self.padding = (self.patch_size - 1) // 2

        assert self.downsample == self.patch_size, f"Downsample should be equal to the patch size."

        gp_resolution = (
            torch.log(torch.tensor(multi_hash_encoder["finest_resolution"], dtype=torch.float32)) -\
            torch.log(torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32))
        ) / (multi_hash_encoder["levels"] - 1)
        resolutions = torch.exp(
            torch.log(
                torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32)
            ).unsqueeze(0) + torch.arange(multi_hash_encoder["levels"]).unsqueeze(1) * gp_resolution.unsqueeze(0)
        ).type(torch.int32).to(device) # since data minibatches are always on GPU
        multi_hash_encoder["resolutions"] = resolutions
        multi_hash_encoder["L_H"] = (torch.log2(resolutions[:,:3] + 1).sum(dim=1) >\
                                     multi_hash_encoder["log2_entries_per_level"]).sum().item()
        multi_hash_encoder["L_NH"] = multi_hash_encoder["levels"] - multi_hash_encoder["L_H"]

        # Multi-resolution hash encoder for the feature field generating events.
        self.logger.info("Instantiating Multi-resolution Hash Encoder with the following resolutions: "+\
                         f"{torch.Tensor(frame_sizes[:3]) / multi_hash_encoder['resolutions'][:, :3].cpu()}")
        self.multi_hash_encoder = MultiResolutionHashEncoder(compile=compile, **multi_hash_encoder)

        in_channels = multi_hash_encoder["feature_size"] * multi_hash_encoder["levels"]

        kernel_strides_dilations = []
        for i in range(self._nstages):
            kernel_strides_dilations.append((dskernels[i], dsstrides[i], 1))
            kernel_strides_dilations.extend([(convkernels[i], 1, convdilations[i]) for _ in range(convdepths[i])])
        spatial_receptive_field = calculate_receptive_field(kernel_strides_dilations)
        self.logger.info(f"Spatial receptive field: {spatial_receptive_field}")

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
            trunc_normal_(m.weight, std=.02)
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
    def init_from_config(cls, eventff_config: str, return_logits: bool=False, return_feat: bool=False,
                         return_loss: bool=False, loss_fn: str="VoxelFocalLoss", **kwargs):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            multi_hash_encoder=conf["multi_hash_encoder"], T = conf["T"], frame_sizes=conf["frame_sizes"],
            dims=conf["dims"], convkernels=conf["convkernels"], convdepths=conf["convdepths"],
            convbtlncks=conf["convbtlncks"], convdilations=conf["convdilations"],
            dskernels=conf["dskernels"], dsstrides=conf["dsstrides"],
            patch_size=conf["patch_size"], use_decoder_block=conf["use_decoder_block"],
            return_logits=return_logits, return_loss=return_loss, return_feat=return_feat,
            loss_fn=loss_fn, device=conf.get("device", "cuda"), variable_mode=conf.get("variable_mode", True),
        )

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

        feature_field = torch.zeros(B, encoded_events.shape[-1], self.w, self.h, device=encoded_events.device, dtype=encoded_events.dtype)
        batch_indices = torch.arange(B).to(encoded_events.device).view(-1, 1).expand(B, N)
        feature_field[batch_indices, :, curr_x, curr_y] += encoded_events

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

        feature_field = torch.zeros((B, encoded_events.shape[-1], self.w, self.h), device=encoded_events.device, dtype=encoded_events.dtype)
        batch_indices = torch.repeat_interleave(eventCounts).int() # Offending line, cant compile. I am not spending time on this for now.
        feature_field[batch_indices, :, curr_x, curr_y] += encoded_events
        return feature_field

    def forward_loss(self, logits: torch.Tensor, futureBlock: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
            logits: torch.Tensor (B,W,H,T)
            futureBlock: torch.Tensor (B,W,H,T)
        """
        loss = self.loss_fn(logits, futureBlock, valid_mask)
        return loss

    def forward(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                futureBlock: torch.Tensor=None, valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B, N, 3/4) or (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,) [N1, N2, N3, ..., NB]
            futureBlock: torch.Tensor (B, W, H, T)
            valid_mask: torch.Tensor (B, W, H)
        """
        if not self.return_logits and not self.return_feat and not self.return_loss:
            return None

        if self.variable_mode:
            x = self.forward_variable(currentBlock, eventCounts) # (B,C,W,H)
        else:
            x = self.forward_fixed(currentBlock) # (B,C,W,H)

        for i in range(self._nstages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if self.return_feat:
            feat = x.permute(0, 2, 3, 1)

        if  not self.return_logits and self.return_feat:
            return None, feat

        if self.use_decoder_block:
            x = self.decoder(x)
        logits = self.pred(x) # (B,C,W,H) -> (B,patch_size**2*T,W/ds,H/ds)

        logits = self.unpatchify(logits) # (B,patch_size**2*T,W/ds,H/ds) -> (B,W,H,T)
        if self.return_loss:
            loss = self.forward_loss(logits, futureBlock, valid_mask)

        ret = (None,)
        if self.return_logits: ret = (logits,)
        if self.return_feat: ret += (feat,)
        if self.return_loss: ret += (loss,)
        return ret


class EventSparsePatchFF(nn.Module):
    """ Sparse ConvNet for Patch prediction
        
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
                 dskernels: list[int]=[5, 5], # Downsample kernel size
                 dsstrides: list[int]=[2, 2], # Downsample stride
                 patch_size=16,
                 use_decoder_block: bool=True,
                 return_feat: bool=False, # Whether to return the key intermediate features or not
                 return_loss: bool=False, # Whether to return the loss or not
                 loss_fn: str="VoxelFocalLoss", # Loss function to use
                 device: str="cuda",
                ):
        super().__init__()
        assert len(dims) == len(convkernels) == len(convdepths) == len(dskernels) == len(dsstrides),\
               f"Length of dims, convkernels, and convdepths should be the same."
        self._nstages = len(dims) 

        self.logger = logging.getLogger("__main__")
        self.frame_sizes = frame_sizes
        self.w, self.h, self.t = frame_sizes

        self.T = T
        self.dims = dims
        self.convkernels = convkernels
        self.convdepths = convdepths
        self.convbtlncks = convbtlncks
        self.convdilations = [1] * len(convdepths) # Sparse convolutions do not use dilation
        self.dskernels = dskernels
        self.dsstrides = dsstrides
        self.patch_size = patch_size
        self.multi_hash_encoder_args = copy(multi_hash_encoder)

        self.use_decoder_block = use_decoder_block
        self.return_feat = return_feat
        self.return_loss = return_loss
        if return_loss:
            self.loss_fn = LOSSES[loss_fn]

        self.downsample = torch.prod(torch.tensor(dsstrides)).item()
        self.padding = (self.patch_size - 1) // 2

        assert self.downsample == self.patch_size, f"Downsample should be equal to the patch size."

        gp_resolution = (
            torch.log(torch.tensor(multi_hash_encoder["finest_resolution"], dtype=torch.float32)) -\
            torch.log(torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32))
        ) / (multi_hash_encoder["levels"] - 1)
        resolutions = torch.exp(
            torch.log(
                torch.tensor(multi_hash_encoder["coarsest_resolution"], dtype=torch.float32)
            ).unsqueeze(0) + torch.arange(multi_hash_encoder["levels"]).unsqueeze(1) * gp_resolution.unsqueeze(0)
        ).type(torch.int32).to(device) # since data minibatches are always on GPU
        multi_hash_encoder["resolutions"] = resolutions
        multi_hash_encoder["L_H"] = (torch.log2(resolutions[:,:3] + 1).sum(dim=1) >\
                                     multi_hash_encoder["log2_entries_per_level"]).sum().item()
        multi_hash_encoder["L_NH"] = multi_hash_encoder["levels"] - multi_hash_encoder["L_H"]

        # Multi-resolution hash encoder for the feature field generating events.
        self.logger.info("Instantiating Multi-resolution Hash Encoder with the following resolutions: "+\
                         f"{torch.Tensor(frame_sizes[:3]) / multi_hash_encoder['resolutions'][:, :3].cpu()}")
        self.multi_hash_encoder = MultiResolutionHashEncoder(compile=compile, **multi_hash_encoder)

        pads = [(dskernels[i]-1)//2 for i in range(self._nstages)]
        in_channels = multi_hash_encoder["feature_size"] * multi_hash_encoder["levels"]

        kernel_strides_dilations = []
        for i in range(self._nstages):
            kernel_strides_dilations.append((dskernels[i], dsstrides[i], 1))
            kernel_strides_dilations.extend([(convkernels[i], 1, 1) for _ in range(convdepths[i])])
        spatial_receptive_field = calculate_receptive_field(kernel_strides_dilations)
        self.logger.info(f"Spatial receptive field: {spatial_receptive_field}")

        self.voxel_grid_pooling = spnn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, self.t), stride=(1, 1, self.t))
        self.voxel_grid_pooling.kernel.data.fill_(1)
        self.voxel_grid_pooling.kernel.requires_grad = False

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            spnn.Conv3d(in_channels, dims[0], kernel_size=(dskernels[0], dskernels[0], 1),
                        stride=(dsstrides[0], dsstrides[0], 1), padding=(pads[0], pads[0], 0), bias=True),
            SparseLayerNorm(dims[0], eps=1e-6)
        )) # collapses the time dimension
        for i in range(1, self._nstages):
            self.downsample_layers.append(nn.Sequential(
                SparseLayerNorm(dims[i-1], eps=1e-6),
                spnn.Conv3d(dims[i-1], dims[i], kernel_size=(dskernels[i], dskernels[i], 1), 
                            stride=dsstrides[i], padding=(pads[i], pads[i], 0), bias=True),
            ))

        self.stages = nn.ModuleList()
        for i in range(self._nstages):
            self.stages.append(nn.Sequential(*[
                SparseBlock(dim=dims[i], kernel_size=convkernels[i], bottleneck=convbtlncks[i]) for _ in range(convdepths[i])
            ]))

        ksize = spatial_receptive_field // patch_size
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        self.decoder = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=ksize, stride=1, padding=(ksize-1)//2, bias=True, groups=dims[-1]),
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(dims[-1], patch_size**2 * T, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (spnn.Conv3d)):
            trunc_normal_(m.kernel, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
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
    def init_from_config(cls, eventff_config: str, return_feat: bool=False,
                         return_loss: bool=False, loss_fn: str="VoxelFocalLoss", **kwargs):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            multi_hash_encoder=conf["multi_hash_encoder"], T = conf["T"], frame_sizes=conf["frame_sizes"],
            dims=conf["dims"], convkernels=conf["convkernels"], convdepths=conf["convdepths"],
            convbtlncks=conf["convbtlncks"], dskernels=conf["dskernels"], dsstrides=conf["dsstrides"],
            patch_size=conf["patch_size"], return_loss=return_loss, return_feat=return_feat,
            loss_fn=loss_fn, device=conf.get("device", "cuda")
        )

    def save_config(self, path: str):
        with open(path, "w") as f:
            yaml.dump({
                "model": "EventSparsePatchFF", "multi_hash_encoder": self.multi_hash_encoder_args, "T": self.T,
                "frame_sizes": self.frame_sizes, "dims": self.dims, "convkernels": self.convkernels,
                "convdepths": self.convdepths, "convbtlncks": self.convbtlncks, "dskernels": self.dskernels,
                "dsstrides": self.dsstrides, "patch_size": self.patch_size
            }, f, default_flow_style=None)

    def forward_variable(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,)    [N1, N2, N3, ..., NB]
        """
        curr_x = (currentBlock[:,0] * self.w).round().int()
        curr_y = (currentBlock[:,1] * self.h).round().int()
        curr_t = (currentBlock[:,2] * self.t).round().int()

        encoded_events = self.multi_hash_encoder(currentBlock.unsqueeze(0)).clone().squeeze(0) # (N,3)/(N,4) -> (N,L*F)
        batch_indices = torch.repeat_interleave(eventCounts) # Offending line, cant compile. I am not spending time on this for now.
        coords = torch.cat([
            batch_indices.unsqueeze(-1), curr_x.unsqueeze(-1), curr_y.unsqueeze(-1), curr_t.unsqueeze(-1)
        ], dim=-1).int()
        return encoded_events, coords

    def forward_loss(self, logits: torch.Tensor, futureBlock: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
            logits: torch.Tensor (B,W,H,T)
            futureBlock: torch.Tensor (B,W,H,T)
        """
        loss = self.loss_fn(logits, futureBlock, valid_mask)
        return loss

    def forward(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                futureBlock: torch.Tensor=None, valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B, N, 3/4) or (N, 3/4) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,) [N1, N2, N3, ..., NB]
            futureBlock: torch.Tensor (B, W, H, T)
            valid_mask: torch.Tensor (B, W, H)
        """
        B = eventCounts.shape[0]
        encoded_events, coords = self.forward_variable(currentBlock, eventCounts) # (B,C,W,H)
        x = SparseTensor(coords=coords, feats=encoded_events).to(currentBlock.device)
        x = self.voxel_grid_pooling(x) # Sums up all events having the same spatial coordinates

        for i in range(self._nstages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        # x.feats = (N', F') x.coords = (N', 4) with x.coords[:, -1] = 0
        # and x.coords[:, 0] = coords[:, 0] (batch indices)
        x = to_dense(x, B, x.feats.shape[-1], self.w // self.patch_size, self.h // self.patch_size)

        if self.use_decoder_block:
            x = self.decoder(x)

        if self.return_feat:
            feat = x.permute(0, 2, 3, 1)
        
        logits = self.pred(x) # (B,C,W,H) -> (B,patch_size**2*T,W/ds,H/ds)
        logits = self.unpatchify(logits) # (B,patch_size**2*T,W/ds,H/ds) -> (B,W,H,T)
        if self.return_loss:
            loss = self.forward_loss(logits, futureBlock, valid_mask)

        ret = (logits,)
        if self.return_feat: ret += (feat,)
        if self.return_loss: ret += (loss,)
        return ret


class EventUNetFF(nn.Module):
    """ Standard UNet for Event prediction using DoubleConv blocks

    Args:
        T (int): Number of frames to predict. Default: 20
        frame_sizes (list[int]): Input size [W, H, T]. Default: [1280, 720, 20]
        dims (list[int]): Feature dimensions at each stage. Default: [32, 64, 128, 256, 512, 1024]
        return_feat (bool): Whether to return the key intermediate features or not. Default: False
        return_loss (bool): Whether to return the loss or not. Default: False
        loss_fn (str): Loss function to use. Default: "EquiWeightedMSELoss"
    """
    def __init__(self,
                 T: int = 20, # Number of frames to predict
                 frame_sizes: list[int] = [1280, 720, 20], # Input size [W, H, T]
                 dims: list[int] = [32, 32], # Feature dimensions at each stage
                 return_logits: bool = False, # Whether to return the logits or not
                 return_feat: bool = False, # Whether to return the key intermediate features or not
                 upsample_returned_feat: bool = True, # Whether to return the features from the upsample path
                 return_loss: bool = False, # Whether to return the loss or not
                 loss_fn: str = "EquiWeightedMSELoss", # Loss function to use
                ):
        super().__init__()
        self.logger = logging.getLogger("__main__")

        self.dims = dims
        self.feature_size = dims[-1]
        self.frame_sizes = frame_sizes
        self.w, self.h, self.t = frame_sizes

        self.inp_t = self.t
        self.pred_t = T

        self.return_logits = return_logits
        self.return_feat = return_feat
        self.return_loss = return_loss
        self.upsample_returned_feat = upsample_returned_feat
        if return_loss:
            self.loss_fn = LOSSES[loss_fn]
            self.logger.info(f"Using loss function: {self.loss_fn}")

        self.ev2voxelgrid = EV2VoxelGrid(self.h, self.w, self.t)

        # Standard U-Net architecture with 6 encoder layers
        self.inc = UNetDoubleConv(self.inp_t, dims[0])  # Initial convolution

        # Encoder (downsampling path) - 6 layers total with 4x total downsampling
        self.down1 = UNetDown(dims[0], dims[0])  # 2x downsample
        self.down2 = UNetDoubleConv(dims[0], dims[1])  # No downsample
        self.down3 = UNetDoubleConv(dims[1], dims[1])  # No downsample
        self.down4 = UNetDown(dims[1], dims[1])  # 2x downsample (total 4x)
        self.down5 = UNetDoubleConv(dims[1], dims[1])  # No downsample
        self.down6 = UNetDoubleConv(dims[1], dims[1])  # No downsample

        # Decoder (upsampling path) - mirror the encoder
        self.up1 = UNetDoubleConv(dims[1] + dims[1], dims[1])  # No upsampling, just combine features
        self.up2 = UNetDoubleConv(dims[1] + dims[1], dims[1])
        self.up3 = UNetUp(dims[1], dims[1], bilinear=False)  # 2x upsample
        self.up4 = UNetDoubleConv(dims[1] + dims[1], dims[0])  # No upsampling, just combine features
        self.up5 = UNetDoubleConv(dims[0] + dims[0], dims[0])
        self.up6 = UNetUp(dims[0], dims[0], bilinear=False)  # 2x upsample

        # Output convolution
        self.pred = nn.Conv2d(dims[0], self.pred_t, kernel_size=1)  # Final prediction
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @classmethod
    def init_from_config(cls, eventff_config: str, return_logits: bool=False, return_feat: bool=False,
                         return_loss: bool=False, loss_fn: str="EquiWeightedMSELoss", **kwargs):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            T=conf["T"], frame_sizes=conf["frame_sizes"], dims=conf["dims"],
            return_logits=return_logits, return_feat=return_feat,
            return_loss=return_loss, loss_fn=loss_fn
        )

    def save_config(self, path: str):
        with open(path, "w") as f:
            yaml.dump({
                "model": "EventUNetFF", "T": self.pred_t,
                "frame_sizes": self.frame_sizes, "dims": self.dims,
            }, f, default_flow_style=None)

    def forward_loss(self, logits: torch.Tensor, futureBlock: torch.Tensor,
                     features: torch.Tensor, valid_mask: torch.Tensor, λ: float=1e-3) -> torch.Tensor:
        """
            logits: torch.Tensor (B,W,H,T)
            futureBlock: torch.Tensor (B,W,H,T)
            features: torch.Tensor (B,W,H,C) / (B,C,W,H) - intermediate features
            valid_mask: torch.Tensor (B, W, H) - mask for valid pixels
            λ: float - regularization weight for features
        """
        return self.loss_fn(logits, futureBlock, valid_mask) + λ * torch.mean(torch.abs(features))

    def forward(self, currentBlock: torch.Tensor, eventCounts: torch.Tensor,
                futureBlock: torch.Tensor=None, valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            currentBlock: torch.Tensor (B, N, 2) or (N, 2) N = N1 + N2 + N3 + ... + NB
            eventCounts: torch.Tensor (B,) [N1, N2, N3, ..., NB]
            futureBlock: torch.Tensor (B, W, H, T)
            valid_mask: torch.Tensor (B, W, H)

            Returns:
                logits: torch.Tensor (B, W, H, T) - predicted future events if return_logits is True
                ret_feat: torch.Tensor (B, W, H, C) - intermediate features if return_feat is True
                loss: torch.Tensor - loss if return_loss is True

            Possible return values:
                None | None, feat | None, feat, loss | logits | logits, feat | logits, feat, loss
        """
        if not self.return_logits and not self.return_feat and not self.return_loss:
            return None

        x = self.ev2voxelgrid(currentBlock, eventCounts).permute(0, 1, 3, 2)  # (B, T, W, H) -> (B, T, W, H)

        # 6-layer encoder with 4x total downsampling
        x1 = self.inc(x)           # Initial convolution              inp_t   -> dims[0]
        x2 = self.down1(x1)        # Down 1 (2x downsample)           dims[0] -> dims[0]
        x3 = self.down2(x2)        # Layer 2 (no downsample)          dims[0] -> dims[1]
        x4 = self.down3(x3)        # Layer 3 (no downsample)          dims[1] -> dims[1]
        x5 = self.down4(x4)        # Down 2 (2x downsample, total 4x) dims[1] -> dims[1]
        x6 = self.down5(x5)        # Layer 5 (no downsample)          dims[1] -> dims[1]
        x_bottom = self.down6(x6)  # Layer 6 (no downsample)          dims[1] -> dims[1]

        # Extract features from the middle layer (right after downsampling ends)
        if self.return_feat:
            if self.upsample_returned_feat:
                ret_feat = F.interpolate(
                    x_bottom, size=(self.w, self.h), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1) # (B, w, h, C)
            else:
                ret_feat = x.permute(0, 2, 3, 1) # (B, W, H, C) - intermediate features without upsampling

        if not self.return_logits and not self.return_loss:
            return None, ret_feat

        if self.return_loss:
            loss_feat = x_bottom.permute(0, 2, 3, 1) # (B, W, H, C) - intermediate features without upsampling

        # 6-layer decoder path with skip connections (mirror of encoder)
        x = self.up1(torch.cat([x6, x_bottom], dim=1))  # No upsampling, just conv
        x = self.up2(torch.cat([x5, x], dim=1))         # No upsampling, just conv
        x = self.up3(x, x4)                             # 2x upsample
        x = self.up4(torch.cat([x3, x], dim=1))         # No upsampling, just conv
        x = self.up5(torch.cat([x2, x], dim=1))         # No upsampling, just conv
        x = self.up6(x, x1)                             # 2x upsample

        # Final prediction
        logits = self.pred(x).permute(0, 2, 3, 1)  # (B, W, H, T)

        ret = (None,)
        if self.return_logits:
            ret = (logits,)
        if self.return_feat:
            ret += (ret_feat,)
        if self.return_loss:
            ret += (self.forward_loss(logits, futureBlock, loss_feat, valid_mask),) # loss
        return ret


def load_weights_ckpt(obj, eventff_ckpt: str, strict: bool=True):
    ckpt_dict = torch.load(eventff_ckpt, weights_only=True)
    obj.load_state_dict(ckpt_dict["model"], strict=strict)
    epoch = ckpt_dict["epoch"]
    loss = ckpt_dict["loss"]
    acc = ckpt_dict["acc"]
    del ckpt_dict
    torch.cuda.empty_cache()
    return epoch, loss, acc


def init_event_model(eventff_config: str, return_feat: bool=False,
                     return_loss: bool=False, loss_fn: str="VoxelFocalLoss", **kwargs) -> nn.Module:
    with open(eventff_config, "r") as f:
        conf = yaml.safe_load(f)
        model = conf["model"]
    if model == "EventPixelFF": model = EventPixelFF
    elif model == "EventPatchFF": model = EventPatchFF
    elif model == "EventSparsePatchFF": model = EventSparsePatchFF
    elif model == "EventUNetFF": model = EventUNetFF
    else: raise ValueError(f"Unknown model: {model}")
    return model.init_from_config(eventff_config, return_feat=return_feat,
                                  return_loss=return_loss, loss_fn=loss_fn, **kwargs)


if __name__ == "__main__":
    # event_ff = EventFF()
    # event_ff = EventPatchFF()
    # event_ff.cuda()
    # x = torch.rand(1,10)
    # y = torch.rand(1,10)
    # t = torch.rand(1,10)
    # fx = torch.rand(1,10)
    # fy = torch.rand(1,10)
    # ft = torch.rand(1,10)
    # event_ff(torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1).cuda(),
    #          torch.cat([fx.unsqueeze(-1), fy.unsqueeze(-1), ft.unsqueeze(-1)], dim=-1).cuda())
    
    event_ff = EventPatchFF(
        multi_hash_encoder={
            "levels": 4, # Number of hash encoding resolutions
            "log2_entries_per_level": 19, # Log2 of the number of entries per level
            "feature_size": 2, # Size of feature vector per entry
            "coarsest_resolution": [8, 8, 1], # Coarsest resolution of the feature field
            "finest_resolution": [320, 180, 8] # Finest resolution of the feature field
        },
        T=20,
        frame_sizes=[1280, 720, 20],
        depths=[2, 2, 2, 2],
        dims=[32, 64, 96, 128],
        patch_size=8
    )
    event_ff.cuda()
    event_ff.train()
    
    event_ff = torch.compile(event_ff, mode="reduce-overhead")
    
    print(num_params(event_ff))
    
    curr_x = torch.randint(0, 1280, (400000,)).cuda() / 1280
    curr_y = torch.randint(0, 720, (400000,)).cuda() / 720
    curr_t = torch.randint(0, 20, (400000,)).cuda() / 20
    currentBlock = torch.cat([curr_x.unsqueeze(-1), curr_y.unsqueeze(-1), curr_t.unsqueeze(-1)], dim=-1).cuda()
    eventCounts = torch.tensor([400000], dtype=torch.int32).cuda()
    
    pred = event_ff(currentBlock, eventCounts)
