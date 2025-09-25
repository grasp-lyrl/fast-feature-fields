import yaml
import logging

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from f3 import init_event_model, load_weights_ckpt
from f3.utils import LayerNorm, Block, EV2Frame, EV2VoxelGrid, batch_cropper, model_to_dict
from f3.tasks.optical_flow.utils import compute_photometric_loss, compute_smoothness_loss, gaussian_kernel2d


class FlowHead(nn.Module):
    def __init__(self,
                 input_channels: int=32,
                 config: dict={},
    ):
        super(FlowHead, self).__init__()
        
        kernels = config.get("kernels", [7, 7, 7, 7])
        btlncks = config.get("btlncks", [2, 2, 2, 2])
        dilations = config.get("dilations", [1, 2, 2, 1])
        
        self.input_channels = input_channels
        decoder_in = self.input_channels

        self.decoder = nn.Sequential(*[
            Block(decoder_in, kernel_size=kernels[i], bottleneck=btlncks[i], dilation=dilations[i])
            for i in range(len(kernels))
        ])
        self.pred = nn.Conv2d(decoder_in, 2, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
            features: (B, C, H, W)
            
            Returns:
                flow: (B, 2, H, W)
        """
        x = self.decoder(features)
        x = self.pred(x)
        return x


class EventFFFlow(nn.Module):
    def __init__(self,
                 eventff_config: str,
                 pyramids: int=4,   # Number of pyramids for the photometric loss
                 alpha: float=1e-3, # Weight for the smoothness loss
                 flowhead_config: dict={},
                 return_loss: bool=False,
    ):
        self.logger = logging.getLogger("__main__")
        super(EventFFFlow, self).__init__()
        
        self.alpha = alpha
        self.pyramids = pyramids
        self.return_loss = return_loss
        self.eventff_config = eventff_config
        self.flowhead_config = flowhead_config
        self.eventff = init_event_model(eventff_config, return_feat=True, return_loss=False).cuda()
        self.flowhead = FlowHead(self.eventff.feature_size, flowhead_config).cuda()
        
        if return_loss:
            self.kernel = gaussian_kernel2d(5, 1.0).cuda()

        # Freeze the entire pretrained EventPixelFF
        for param in self.eventff.parameters():
            param.requires_grad = False

    def save_configs(self, path: str):
        with open(path + "/flow_config.yaml", "w") as yaml_file:
            yaml.dump({
                "model": "EventFFFlow", "eventff": self.eventff_config, "flowhead": self.flowhead_config
            }, yaml_file, default_flow_style=None)
        
        model_architecture = model_to_dict(self)
        with open(path + "/model_architecture.yaml", "w") as yaml_file:
            yaml.dump(model_architecture, yaml_file, default_flow_style=False)

    @classmethod
    def init_from_config(cls, path: str, pyramids: int=4, alpha: float=1e-3, return_loss: bool=False):
        with open(path, "r") as yaml_file:
            flow_config = yaml.safe_load(yaml_file)
        return cls(
            eventff_config=flow_config["eventff"],
            pyramids=pyramids,
            alpha=alpha,
            flowhead_config=flow_config["flowhead"],
            return_loss=return_loss
        )

    def load_weights(self, eventff_ckpt: str):
        epoch, loss, acc = load_weights_ckpt(self.eventff, eventff_ckpt, strict=False)
        self.logger.info(f"Loaded EventPixelFF ckpt from {eventff_ckpt}. " +\
                         f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}")

    def forward_loss(self, image1: torch.Tensor, image2: torch.Tensor, flow: torch.Tensor,
                     valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            image1: (B, C, H, W)
            image2: (B, C, H, W)
            flow: (B, 2, H, W)
            valid_mask: (B, 1, H, W)
        """
        photometric_loss = compute_photometric_loss(image1, image2, flow, valid_mask,
                                                    pyr_levels=self.pyramids, kernel=self.kernel)
        smoothness_loss = compute_smoothness_loss(flow, pyr_levels=self.pyramids)
        self.logger.info(f"Photometric Loss: {photometric_loss.item()}, " +\
                         f"Smoothness Loss: {smoothness_loss.item()}")
        loss = photometric_loss + self.alpha * smoothness_loss
        return {
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "loss": loss
        }

    def forward(self,
                eventBlockFlow: torch.Tensor, eventCountsFlow: torch.Tensor,
                eventBlockOld: torch.Tensor=None, eventCountsOld: torch.Tensor=None,
                eventBlockNew: torch.Tensor=None, eventCountsNew: torch.Tensor=None,
                cparams: torch.Tensor=None) -> torch.Tensor:
        """
            eventBlockFlow: (B, N, 4) # Events used to predict the flow
            eventCountsFlow: (B, 1)
            eventBlockOld: (B, N, 4)
            eventCountsOld: (B, 1)
            eventBlockNew: (B, N, 4)
            eventCountsNew: (B, 1)
            cparams: (B, 4)
        """
        with torch.no_grad():
            ffflow = self.eventff(eventBlockFlow, eventCountsFlow)[1].permute(0, 3, 2, 1) # (B, C, H, W)
            ffflow = batch_cropper(ffflow, cparams)
        flow_pred = self.flowhead(ffflow)

        if self.return_loss:
            with torch.no_grad():
                ff1 = self.eventff(eventBlockOld, eventCountsOld)[1].permute(0, 3, 2, 1) # (B, C, H, W)
                ff2 = self.eventff(eventBlockNew, eventCountsNew)[1].permute(0, 3, 2, 1) # (B, C, H, W)
                ff1 = batch_cropper(ff1, cparams)
                ff2 = batch_cropper(ff2, cparams)
            loss = self.forward_loss(ff1, ff2, flow_pred, None)

        ret = (flow_pred, ffflow)
        if self.return_loss:
            ret += (ff1, ff2, loss,)
        return ret


class EventFlow(nn.Module):
    def __init__(self,
                 eventmodel: str="frames",
                 width: int=1280,
                 height: int=720,
                 timebins: int=20,
                 pyramids: int=4,   # Number of pyramids for the photometric loss
                 alpha: float=1e-3, # Weight for the smoothness loss
                 flowhead_config: dict={},
                 return_loss: bool=False,
    ):
        self.logger = logging.getLogger("__main__")
        super(EventFlow, self).__init__()

        if eventmodel == "frames":
            evmodel = EV2Frame(height, width, polarity=True)
        elif eventmodel == "voxelgrid":
            evmodel = EV2VoxelGrid(height, width, timebins)
        else:
            raise NotImplementedError(f"Event model {eventmodel} not implemented")

        self.alpha = alpha
        self.width = width
        self.height = height
        self.timebins = timebins
        self.pyramids = pyramids
        self.eventmodel = evmodel
        self.return_loss = return_loss
        self.eventmodel_name = eventmodel
        self.flowhead_config = flowhead_config
        self.normalize = (eventmodel != "frames")

        #! For a fair comparison we restrict the input channels to 32.
        #! That is upsample the number of channels to be the same as the FF,
        #! ensuring similar number of parameters in the flowhead...
        self.flowhead = FlowHead(32, flowhead_config).cuda()
        self.upchannel = nn.Sequential(
            nn.Conv2d(self.eventmodel._chan, self.flowhead.input_channels, kernel_size=1),
            LayerNorm(self.flowhead.input_channels, eps=1e-6, data_format="channels_first")
        ).cuda()

        if return_loss:
            self.kernel = gaussian_kernel2d(5, 1.0).cuda()

    def save_configs(self, path: str):
        with open(path + "/flow_config.yaml", "w") as yaml_file:
            yaml.dump({
                "model": "EventFlow", "eventmodel": self.eventmodel_name, "width": self.width,
                "height": self.height, "timebins": self.timebins, "flowhead": self.flowhead_config
            }, yaml_file, default_flow_style=None)

        model_architecture = model_to_dict(self)
        with open(path + "/model_architecture.yaml", "w") as yaml_file:
            yaml.dump(model_architecture, yaml_file, default_flow_style=False)

    @classmethod
    def init_from_config(cls, path: str, pyramids: int=4, alpha: float=1e-3, return_loss: bool=False):
        with open(path, "r") as yaml_file:
            flow_config = yaml.safe_load(yaml_file)
        return cls(
            eventmodel=flow_config["eventmodel"],
            width=flow_config["width"],
            height=flow_config["height"],
            timebins=flow_config["timebins"],
            pyramids=pyramids,
            alpha=alpha,
            flowhead_config=flow_config["flowhead"],
            return_loss=return_loss
        )

    def forward_loss(self, image1: torch.Tensor, image2: torch.Tensor, flow: torch.Tensor,
                     valid_mask: torch.Tensor=None) -> torch.Tensor:
        """
            image1: (B, C, H, W)
            image2: (B, C, H, W)
            flow: (B, 2, H, W)
            valid_mask: (B, 1, H, W)
        """
        photometric_loss = compute_photometric_loss(image1, image2, flow, valid_mask,
                                                    self.pyramids, self.kernel, self.normalize)
        smoothness_loss = compute_smoothness_loss(flow, self.pyramids)
        self.logger.info(f"Photometric Loss: {photometric_loss.item()}, " +\
                         f"Smoothness Loss: {smoothness_loss.item()}")
        loss = photometric_loss + self.alpha * smoothness_loss
        return {
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "loss": loss
        }

    def forward(self,
                eventBlockFlow: torch.Tensor, eventCountsFlow: torch.Tensor,
                eventBlockOld: torch.Tensor=None, eventCountsOld: torch.Tensor=None,
                eventBlockNew: torch.Tensor=None, eventCountsNew: torch.Tensor=None,
                cparams: torch.Tensor=None) -> torch.Tensor:
        """
            eventBlockFlow: (B, N, 4) # Events used to predict the flow
            eventCountsFlow: (B, 1)
            eventBlockOld: (B, N, 4)
            eventCountsOld: (B, 1)
            eventBlockNew: (B, N, 4)
            eventCountsNew: (B, 1)
            cparams: (B, 4)
        """
        with torch.no_grad():
            ffflow = self.eventmodel(eventBlockFlow, eventCountsFlow) # (B, C, H, W)
            ffflow = batch_cropper(ffflow, cparams)
        flow_pred = self.flowhead(self.upchannel(ffflow))

        if self.return_loss:
            with torch.no_grad():
                ff1 = self.eventmodel(eventBlockOld, eventCountsOld) # (B, C, H, W)
                ff2 = self.eventmodel(eventBlockNew, eventCountsNew) # (B, C, H, W)
                ff1 = batch_cropper(ff1, cparams)
                ff2 = batch_cropper(ff2, cparams)
            loss = self.forward_loss(ff1, ff2, flow_pred, None)

        ret = (flow_pred, ffflow)
        if self.return_loss:
            ret += (ff1, ff2, loss,)
        return ret


def init_flow_model(path: str, pyramids: int=4, alpha: float=1e-3, return_loss: bool=False) -> nn.Module:
    """
        Initialize the flow model from the config file.
        Args:
            path (str): Path to the config file.
            pyramids (int): Number of pyramids for the photometric loss.
            alpha (float): Weight for the smoothness loss.
            return_loss (bool): Whether to return the loss or not.
        Returns:
            nn.Module: Initialized flow model.
    """
    with open(path, "r") as yaml_file:
        flow_config = yaml.safe_load(yaml_file)
        model = flow_config["model"]
    if model == "EventFFFlow": model = EventFFFlow
    elif model == "EventFlow": model = EventFlow
    else: raise ValueError(f"Unknown flow model: {model}")
    return model.init_from_config(path, pyramids=pyramids, alpha=alpha, return_loss=return_loss)


def load_flow_weights(model: nn.Module, path: str, strict: bool=True) -> None:
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch = ckpt.get("epoch", None)
    loss = ckpt.get("loss", None)
    del ckpt
    torch.cuda.empty_cache()
    return epoch, loss
