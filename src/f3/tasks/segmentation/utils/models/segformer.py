import yaml
import logging

import torch
import torch.nn as nn

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from f3 import init_event_model, load_weights_ckpt
from f3.utils import model_to_dict, EV2Frame, EV2VoxelGrid, batch_cropper


class FFSegformer(nn.Module):
    def __init__(self,
                 segformer_config: str="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                 num_labels: int=11,
                 input_channels: int=32
    ):
        super(FFSegformer, self).__init__()
        self.logger = logging.getLogger("__main__")

        INPUT_CHANNELS = input_channels

        self.config = SegformerConfig.from_pretrained(segformer_config)
        self.config.num_channels = INPUT_CHANNELS

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_config) # (C, H, W)
        #! modify the number of input channels to be 32
        #! maybe theres a better way to do this
        if input_channels != 3:
            old_weights = self.segformer.segformer.encoder.patch_embeddings[0].proj.weight # [384, 3, 14, 14]
            old_bias = self.segformer.segformer.encoder.patch_embeddings[0].proj.bias

            self.segformer.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
                in_channels=INPUT_CHANNELS,
                out_channels=self.segformer.segformer.encoder.patch_embeddings[0].proj.out_channels,
                kernel_size=self.segformer.segformer.encoder.patch_embeddings[0].proj.kernel_size,
                stride=self.segformer.segformer.encoder.patch_embeddings[0].proj.stride,
                padding=self.segformer.segformer.encoder.patch_embeddings[0].proj.padding,
                bias=self.segformer.segformer.encoder.patch_embeddings[0].proj.bias is not None
            )

            new_weights = torch.cat(
                [old_weights.repeat(1, INPUT_CHANNELS // 3, 1, 1), old_weights[:, :INPUT_CHANNELS % 3]],
                dim=1
            )
            self.segformer.segformer.encoder.patch_embeddings[0].proj.weight.data = new_weights
            self.segformer.segformer.encoder.patch_embeddings[0].proj.bias.data = old_bias

        if num_labels < 19:
            if num_labels == 11:
                #! Just load the relevant weights when cityscapes is subsampled to 11 classes.
                indices = torch.tensor([10, 2, 4, 11, 5, 0, 1, 8, 13, 3, 7], dtype=torch.int32)
            else:
                indices = torch.arange(num_labels, dtype=torch.int32)
            old_weights = self.segformer.decode_head.classifier.weight
            old_bias = self.segformer.decode_head.classifier.bias

            self.segformer.decode_head.classifier = nn.Conv2d(
                in_channels=self.segformer.decode_head.classifier.in_channels,
                out_channels=num_labels,
                kernel_size=self.segformer.decode_head.classifier.kernel_size,
                stride=self.segformer.decode_head.classifier.stride,
                padding=self.segformer.decode_head.classifier.padding,
                bias=self.segformer.decode_head.classifier.bias is not None
            )

            self.segformer.decode_head.classifier.weight.data = old_weights[indices]
            self.segformer.decode_head.classifier.bias.data = old_bias[indices]
        self.segformer.cuda()

    def save_configs(self, path: str):
        config_dict = self.config.to_dict()
        with open(path + "/segformer_config.yaml", "w") as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False)

        model_architecture = model_to_dict(self)
        with open(path + "/model_architecture.yaml", "w") as yaml_file:
            yaml.dump(model_architecture, yaml_file, default_flow_style=False)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        segmap = self.segformer(features)['logits']
        upsampled_logits = nn.functional.interpolate(
            segmap, size=features.shape[-2:], mode='bilinear', align_corners=False
        )
        return upsampled_logits


class EventFFSegformer(FFSegformer):
    def __init__(self,
                 eventff_config: str="/home/richeek/GitHub/f3/outputs/ff_9_1_fullcar_leftright_var/models/config.yml",
                 segformer_config: str="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                 num_labels: int=11, # Since most event datasets have 11 classes
    ):
        eventff = init_event_model(eventff_config, return_feat=True, return_loss=False).cuda()
        super(EventFFSegformer, self).__init__(segformer_config, num_labels, eventff.feature_size)

        self.eventff = eventff
        self.num_labels = num_labels
        self.eventff_config = eventff_config
        self.segformer_config = segformer_config
        # Freeze the entire pretrained EventPixelFF
        for param in self.eventff.parameters():
            param.requires_grad = False

    def load_weights(self, eventff_ckpt: str):
        epoch, loss, acc = load_weights_ckpt(self.eventff, eventff_ckpt, strict=False)
        self.logger.info(f"Loaded EventPixelFF ckpt from {eventff_ckpt}. " +\
                         f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}")

    def save_configs(self, path: str):
        loading_facilities = {
            "model": "EventFFSegformer",
            "num_labels": self.num_labels,
            "eventff_config": self.eventff_config,
            "segformer_config": self.segformer_config,
        }
        with open(path + "/segmentation_config.yaml", "w") as yaml_file:
            yaml.dump(loading_facilities, yaml_file, default_flow_style=False)
        super().save_configs(path)

    @classmethod
    def init_from_config(cls, path: str):
        with open(path, "r") as yaml_file:
            loading_facilities = yaml.safe_load(yaml_file)
        return cls(
            segformer_config=loading_facilities["segformer_config"],
            num_labels=loading_facilities["num_labels"],
            eventff_config=loading_facilities["eventff_config"]
        )

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                cparams: torch.Tensor=torch.tensor([[0, 0, 720, 1280]], dtype=torch.int32)) -> torch.Tensor:
        ff = self.eventff(eventBlock, eventCounts)[1].permute(0, 3, 2, 1) # (B, C, H, W)
        ff = batch_cropper(ff, cparams)
        return super().forward(ff), ff


class EventSegformer(FFSegformer):
    def __init__(self,
                 segformer_config: str="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                 num_labels: int=11,
                 eventmodel: str="frames",
                 width: int=1280,
                 height: int=720,
                 timebins: int=20
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
        self.num_labels = num_labels
        self.eventmodel_name = eventmodel
        self.segformer_config = segformer_config

        super(EventSegformer, self).__init__(segformer_config, num_labels, evmodel._chan)
        self.eventmodel = evmodel
    
    def save_configs(self, path):
        loading_facilities = {
            "model": "EventSegformer",
            "num_labels": self.num_labels,
            "eventmodel": self.eventmodel_name,
            "segformer_config": self.segformer_config,
            "width": self.width,
            "height": self.height,
            "timebins": self.timebins
        }
        with open(path + "/segmentation_config.yaml", "w") as yaml_file:
            yaml.dump(loading_facilities, yaml_file, default_flow_style=False)
        super().save_configs(path)

    @classmethod
    def init_from_config(cls, path: str):
        with open(path, "r") as yaml_file:
            loading_facilities = yaml.safe_load(yaml_file)
        return cls(
            segformer_config=loading_facilities["segformer_config"],
            num_labels=loading_facilities["num_labels"],
            eventmodel=loading_facilities["eventmodel"],
            width=loading_facilities["width"],
            height=loading_facilities["height"],
            timebins=loading_facilities["timebins"]
        )

    def forward(self, eventBlock: torch.Tensor, eventCounts: torch.Tensor=None,
                cparams: torch.Tensor=torch.tensor([0, 0, 720, 1280], dtype=torch.int32)) -> torch.Tensor:
        ff = self.eventmodel(eventBlock, eventCounts) # (B, C, H, W)
        ff = batch_cropper(ff, cparams)
        return super().forward(ff), ff


def init_segmentation_model(path: str):
    """
        Initialize the segmentation model from the loading facilities yaml file.
    """
    with open(path, "r") as yaml_file:
        loading_facilities = yaml.safe_load(yaml_file)
        model = loading_facilities["model"]
    if model == "EventSegformer": model = EventSegformer
    elif model == "EventFFSegformer": model = EventFFSegformer
    else: raise ValueError(f"Unknown model {model} in loading facilities.")
    return model.init_from_config(path)


def load_segmentation_weights(model: nn.Module, path: str, strict: bool=True) -> nn.Module:
    """
        Load the weights of the segmentation model from the given path.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch, loss, acc, miou = ckpt["epoch"], ckpt["loss"], ckpt["acc"], ckpt["miou"]
    del ckpt
    torch.cuda.empty_cache()
    return epoch, loss, acc, miou
