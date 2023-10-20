"""
Altered backbone
Alter the backbone input channel
"""
from . import resnet
import torch
import torch.nn.functional as F
import torchvision
from .position_encoding import build_position_encoding, build_position_encoding_shape
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torch import nn

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()

        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        out = {}
        for name, x in xs.items():
            out[name] = x
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 pretained:bool,
                 input_channels,
                 strides,
                 inplanes,
                 channels
                 ):
        backbone = getattr(resnet, name)(
            replace_stride_with_dilation=[dilation, dilation, dilation],
            pretrained=pretained, norm_layer=FrozenBatchNorm2d, input_channels=input_channels, strides=strides, inplanes=inplanes, layer_channels=channels)
        num_channels = channels[-1]

        super().__init__(backbone, train_backbone, num_channels)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():

            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return out, pos



def build_backbone(args):
    position_embedding = build_position_encoding(args, args.pos_code_len_node)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation, args.pretrained_backbone,args.Tbackbone_input_channels ,args.Tbackbone_strides,args.Tbackbone_inplanes ,args.Tbackbone_channels)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_backbone_etfr(args):
    """
    :rtype: object
    """
    position_embedding = build_position_encoding_shape(args, args.pos_code_len_edge, args.edge_pos_shape)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation, args.pretrained_backbone,args.bin_backbone_input_channels ,args.bin_backbone_strides,args.bin_backbone_inplanes ,args.bin_backbone_channels)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model



