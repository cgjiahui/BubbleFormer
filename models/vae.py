import torch
import torch.nn.functional as F
import torchvision
from .position_encoding import build_position_encoding
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torch import nn
from . import resnet

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

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class CnetBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool):
        super().__init__()
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        out = {}

        for name, x in xs.items():
            out[name] = x
        return out

class Cnet(CnetBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 pretained:bool,
                 channels:List[int],
                 c_input_channels:int
                 ):
        backbone = getattr(resnet, name)(
            replace_stride_with_dilation=[dilation, dilation, dilation],
            pretrained=pretained, norm_layer=FrozenBatchNorm2d, layer_channels=channels, input_channels=c_input_channels)
        super().__init__(backbone, train_backbone)



class EnetBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, layer_channels, noise_channels):
        super().__init__()

        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.decoder1=nn.Conv2d(layer_channels[-1],noise_channels,1)
        self.decoder2=nn.Conv2d(layer_channels[-1],noise_channels,1)

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        out = {}

        for name, x in xs.items():
            out[name] = x
            out1 = self.decoder1(x)
            out2 = self.decoder2(x)
        return out1,out2


class Enet(EnetBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 pretained: bool,
                 channels: List[int],
                 e_input_channels:int,
                 e_inplanes:int,
                 e_strides:List[int],
                 noise_channels: int
                 ):
        backbone = getattr(resnet, name)(
            replace_stride_with_dilation=[dilation, dilation, dilation],
            pretrained=pretained, norm_layer=FrozenBatchNorm2d, layer_channels=channels, input_channels=e_input_channels, inplanes=e_inplanes, strides=e_strides)
        super().__init__(backbone, train_backbone, channels, noise_channels)



def build_vae(args):
    cnet=Cnet(args.c_backbone, args.c_train_backbone, args.c_dilation, args.c_pretrained, args.c_channels, args.c_input_channels)
    enet=Enet(args.e_backbone, args.e_train_backbone, args.e_dilation, args.e_pretrained, args.e_channels, args.e_input_channels, args.e_inplanes, args.e_strides, args.noise_channels)
    return cnet,enet


