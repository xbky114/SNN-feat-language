from .fcos import FCOS
from .retinanet import RetinaNet,RetinaNetHead
from .resnet import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torch import nn, Tensor
from functools import partial
from typing import Optional,Any,List,Tuple
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from torchvision.models.detection.anchor_utils import AnchorGenerator

class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.relu = nn.ReLU()
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels
    def forward(
        self,
        p: List[Tensor],
        c: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(self.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names
    
def fcos_resnet50_fpn(
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> FCOS:
    num_classes = 91
    norm_layer = nn.BatchNorm2d
    backbone = resnet50(norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(
        backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = FCOS(backbone, num_classes, **kwargs)
    return model


def retinanet_resnet50_fpn(
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> RetinaNet:
    num_classes = 91
    norm_layer = nn.BatchNorm2d
    backbone = resnet50(norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(
        backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes, **kwargs)
    return model

def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator

def retinanet_resnet50_fpn_v2(
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> RetinaNet:
    
    num_classes = 91
    backbone = resnet50()
    backbone = _resnet_fpn_extractor(
        backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256)
    )
    
    anchor_generator = _default_anchorgen()
    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"
    
    model = RetinaNet(backbone, num_classes, anchor_generator=anchor_generator, head=head, **kwargs)
    return model
