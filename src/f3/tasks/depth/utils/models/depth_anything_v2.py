import yaml
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from f3 import init_event_model, load_weights_ckpt
from f3.tasks.depth.utils import get_resize_shapes
from f3.utils import model_to_dict, EV2Frame, EV2VoxelGrid, batch_cropper
from f3.tasks.depth.depth_anything_v2.dpt import DepthAnythingV2


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class FFDepthAnythingV2(nn.Module):
    def __init__(self,
                 dav2_config: dict,
                 input_channels: int=32
    ):
        super(FFDepthAnythingV2, self).__init__()
        self.logger = logging.getLogger("__main__")

        self.dav2 = DepthAnythingV2(**model_configs[dav2_config['encoder']])

        if 'ckpt' in dav2_config:
            self.dav2.load_state_dict(torch.load(dav2_config['ckpt'], map_location="cpu", weights_only=True))
            self.logger.info(f"Loaded DepthAnythingV2 ckpt from {dav2_config['ckpt']}")

        if input_channels != 3:
            old_weights = self.dav2.pretrained.patch_embed.proj.weight # [384, 3, 14, 14]
            old_bias = self.dav2.pretrained.patch_embed.proj.bias      # [384]
            
            self.dav2.pretrained.patch_embed.proj = nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.dav2.pretrained.patch_embed.proj.out_channels,
                kernel_size=self.dav2.pretrained.patch_embed.proj.kernel_size,
                stride=self.dav2.pretrained.patch_embed.proj.stride,
                padding=self.dav2.pretrained.patch_embed.proj.padding,
                bias=self.dav2.pretrained.patch_embed.proj.bias is not None
            )
            new_weights = torch.cat(
                [old_weights.repeat(1, input_channels // 3, 1, 1), old_weights[:, :input_channels % 3]],
                dim=1
            )
            self.dav2.pretrained.patch_embed.proj.weight.data = new_weights
            self.dav2.pretrained.patch_embed.proj.bias.data = old_bias
        self.dav2.cuda()

    def save_configs(self, path: str):
        model_architecture = model_to_dict(self)
        with open(path + "/model_architecture.yml", "w") as yaml_file:
            yaml.dump(model_architecture, yaml_file)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dav2(features)


class EventFFDepthAnythingV2(FFDepthAnythingV2):
    def __init__(self,
                 eventff_config: str="outputs/pixel_ftft_9_1_focal_fb_400k_20ms_fullcarseq/args.yml",
                 dav2_config: dict=None,
                 retrain: bool=False,
    ):
        self.dav2_config = dav2_config
        self.eventff_config = eventff_config
        eventff = init_event_model(eventff_config, return_feat=True, return_loss=False).cuda()
        super(EventFFDepthAnythingV2, self).__init__(dav2_config, eventff.feature_size)

        self.eventff = eventff
        self.size = dav2_config['size']
        if not retrain:
            for param in self.eventff.parameters():
                param.requires_grad = False

    def load_weights(self, eventff_ckpt: str):
        epoch, loss, acc = load_weights_ckpt(self.eventff, eventff_ckpt, strict=False)
        self.logger.info(f"Loaded EventPixelFF ckpt from {eventff_ckpt}. " +\
                         f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}")

    def save_configs(self, path: str):
        loading_facilities = {
            "model": "EventFFDepthAnythingV2",
            "dav2_config": self.dav2_config,
            "eventff_config": self.eventff_config,
        }
        with open(path + "/depth_config.yml", "w") as yaml_file:
            yaml.dump(loading_facilities, yaml_file, default_flow_style=None)
        super().save_configs(path)

    @classmethod
    def init_from_config(cls, path: str, retrain: bool=False):
        with open(path, "r") as yaml_file:
            loading_facilities = yaml.safe_load(yaml_file)
        return cls(
            eventff_config=loading_facilities['eventff_config'],
            dav2_config=loading_facilities['dav2_config'],
            retrain=retrain
        )

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                cparams: torch.Tensor=torch.tensor([0, 0, 720, 1280], dtype=torch.int32)) -> torch.Tensor:
        ff = self.eventff(eventBlock, eventCounts)[1].permute(0, 3, 2, 1) # (B, C, H, W)
        # All of them should be the same sized crops for training and divisible by 14
        ff = batch_cropper(ff, cparams) # (B, C, H', H')
        ff_interp = F.interpolate(ff, (self.size, self.size), mode='bilinear', align_corners=False)
        depth_pred = super().forward(ff_interp).unsqueeze(1) # (B, 1, size, size)
        depth_pred = F.interpolate(depth_pred, ff.shape[2:], mode="bilinear", align_corners=True).squeeze(1) # (B, H', H')
        return depth_pred, ff # (B, H', H'), (B, C, H', H')

    @torch.no_grad()
    def infer_image(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                    cparams: torch.Tensor=torch.tensor([0, 0, 720, 1280], dtype=torch.int32)) -> torch.Tensor:
        """
            eventBlock: (N,3)
            eventCounts: (1,)
            cparams: (4,) [ofstx, ofsty, h, w]
        """
        # Get the resized shapes if the smaller edge is resized to 518 keeping aspect ratio same
        h, w = cparams[2] - cparams[0], cparams[3] - cparams[1]
        fh, fw = get_resize_shapes(h, w, self.size, 14) # 720,1280 -> 518,924 & 480,640 -> 518, 700

        ff = self.eventff(eventBlock, eventCounts)[1].permute(0, 3, 2, 1) # (1, C, H, W)
        ff = ff[0, :, cparams[0]:cparams[2], cparams[1]:cparams[3]]
        ff_interp = F.interpolate(ff.unsqueeze(0), (fh, fw), mode='bilinear', align_corners=False)

        # Get the depth prediction
        depth_pred = self.dav2(ff_interp).squeeze(1)
        depth_pred = F.interpolate(depth_pred.unsqueeze(0), (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth_pred, ff # (H, W), (C, H, W)


def get_models(eventff_config: str, dav2_config: dict, retrain: bool=False, return_rgb_model: bool=False):
    event_dav2 = EventFFDepthAnythingV2(eventff_config, dav2_config, retrain)
    if return_rgb_model:
        rgb_model = FFDepthAnythingV2(dav2_config, input_channels=3, freeze=True)
        rgb_model.eval()
        return event_dav2, rgb_model
    return event_dav2


class EventDepthAnythingV2(FFDepthAnythingV2):
    def __init__(self,
                 dav2_config: dict,
                 eventmodel: str="frames",
                 width: int=1280,
                 height: int=720,
                 timebins: int=20,
    ):
        if eventmodel == "frames":
            evmodel = EV2Frame(height, width)
        elif eventmodel == "voxelgrid":
            evmodel = EV2VoxelGrid(height, width, timebins)
        else:
            raise NotImplementedError(f"Event model {eventmodel} not implemented")

        self.width = width
        self.height = height
        self.timebins = timebins
        self.dav2_config = dav2_config
        self.eventmodel_name =  eventmodel

        super(EventDepthAnythingV2, self).__init__(dav2_config, input_channels=evmodel._chan)
        self.eventmodel = evmodel
        self.size = dav2_config['size']

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                cparams: torch.Tensor=torch.tensor([0, 0, 720, 1280], dtype=torch.int32)) -> torch.Tensor:
        ff = self.eventmodel(eventBlock, eventCounts) # (B, C, H, W)
        ff = batch_cropper(ff, cparams) # (B, C, H', H')
        ff_interp = F.interpolate(ff, (self.size, self.size), mode='bilinear', align_corners=False)
        depth_pred = super().forward(ff_interp).unsqueeze(1) # (B, 1, size, size)
        depth_pred = F.interpolate(depth_pred, ff.shape[2:], mode="bilinear", align_corners=True).squeeze(1) # (B, H', H')
        return depth_pred, ff # (B, H', H'), (B, C, H', H')
    
    def save_configs(self, path: str):
        loading_facilities = {
            "model": "EventDepthAnythingV2",
            "dav2_config": self.dav2_config,
            "eventmodel": self.eventmodel_name,
            "width": self.width,
            "height": self.height,
            "timebins": self.timebins,
        }
        with open(path + "/depth_config.yml", "w") as yaml_file:
            yaml.dump(loading_facilities, yaml_file)
        super().save_configs(path)
    
    @classmethod
    def init_from_config(cls, path: str, **kwargs):
        with open(path, "r") as yaml_file:
            loading_facilities = yaml.safe_load(yaml_file)
        return cls(
            dav2_config=loading_facilities['dav2_config'],
            eventmodel=loading_facilities['eventmodel'],
            width=loading_facilities['width'],
            height=loading_facilities['height'],
            timebins=loading_facilities['timebins'],
        )
    
    @torch.no_grad()
    def infer_image(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                    cparams: torch.Tensor=torch.tensor([0, 0, 720, 1280], dtype=torch.int32)) -> torch.Tensor:
        """
            eventBlock: (N,3)
            eventCounts: (1,)
            cparams: (4,) [ofstx, ofsty, h, w]
        """
        # Get the resized shapes if the smaller edge is resized to 518 keeping aspect ratio same
        h, w = cparams[2] - cparams[0], cparams[3] - cparams[1]
        fh, fw = get_resize_shapes(h, w, self.size, 14) # 720,1280 -> 518,924 & 480,640 -> 518, 700

        ff = self.eventmodel(eventBlock, eventCounts)
        ff = ff[0, :, cparams[0]:cparams[2], cparams[1]:cparams[3]]
        ff_interp = F.interpolate(ff.unsqueeze(0), (fh, fw), mode='bilinear', align_corners=False)

        # Get the depth prediction
        depth_pred = self.dav2(ff_interp).squeeze(1)
        depth_pred = F.interpolate(depth_pred.unsqueeze(0), (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth_pred, ff # (H, W), (C, H, W)


def init_depth_model(path: str, retrain: bool=False):
    """
        path: path to the loading facilities yaml file
    """
    with open(path, "r") as yaml_file:
        loading_facilities = yaml.safe_load(yaml_file)
        model = loading_facilities.get('model', None)
    if model == "EventFFDepthAnythingV2": model = EventFFDepthAnythingV2
    elif model == "EventDepthAnythingV2": model = EventDepthAnythingV2
    else: raise ValueError(f"Unknown model {model} in loading facilities {loading_facilities}")
    return model.init_from_config(path, retrain=retrain)


def load_depth_weights(model: nn.Module, path: str, strict: bool=True):
    """
        Load the weights of the model from the given path.
    """
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt['model'], strict=strict)
    epoch = ckpt.get('epoch', None)
    results = ckpt.get('results', None)
    del ckpt
    torch.cuda.empty_cache()
    return epoch, results
