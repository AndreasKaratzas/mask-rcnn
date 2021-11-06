import torch

from typing import Tuple, List

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone

from lib.utils import list2tup, flatten


def compile_backbone(
        backbone_name: str = 'resnet50'
) -> torch.nn.Module:
    """Returns the backbone model."""
    if 'resnet' in backbone_name:
        backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
    elif 'mobilenet' in backbone_name:
        backbone = mobilenet_backbone(backbone_name, pretrained=True, fpn=True)
    else:
        raise ValueError(
            f'Backbone model name option is invalid. Input value was {backbone_name}')
    return backbone


def compile_anchor(
        anchor_sizes: Tuple[Tuple[int]] = None, aspect_ratios: Tuple[Tuple[int]] = None
) -> AnchorGenerator:
    """Returns the anchor generator."""
    if anchor_sizes is None:
        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))

    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    else:
        aspect_ratios = ((tuple(flatten(aspect_ratios)),) * len(anchor_sizes))

    return AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)


def compile_rpn(
        in_channels: int, num_anchors: int
) -> RPNHead:
    """Returns the RPN Head"""
    return RPNHead(in_channels=in_channels, num_anchors=num_anchors)


def compile_fast_roi(
        featmap_names: List[str] = ["0"], output_size: int = 7, sampling_ratio: int = 2
) -> MultiScaleRoIAlign:
    """Returns the R-CNN ROI Pooler"""
    return MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio)


def compile_mask_roi(
    featmap_names: List[str] = ["0"], output_size: int = 14, sampling_ratio: int = 2
) -> MultiScaleRoIAlign:
    """Returns the Mask ROI Pooler"""
    return MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio)


def configure_model(
        # Fast R-CNN model parameters
        backbone_name: str,
        anchor_sizes: List[int],
        aspect_ratios: List[int],
        in_channels: int = 256,
        featmap_names: List[str] = ['0'],
        output_size: int = 7,
        sampling_ratio: int = 2,
        representation_size: int = 1024,
        num_classes: int = 2,
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: List[int] = [0.485, 0.456, 0.406],
        image_std: List[int] = [0.229, 0.224, 0.225],
        # Mask R-CNN model parameters
        mask_featmap_names: List[str] = ['0'],
        mask_output_size: int = 14,
        mask_sampling_ratio: int = 2
) -> MaskRCNN:
    """Driver definition for Mask R-CNN model compilation"""
    backbone = compile_backbone(backbone_name=backbone_name)

    rpn_anchor_generator = compile_anchor(
        anchor_sizes=list2tup(anchor_sizes),
        aspect_ratios=list2tup(aspect_ratios)
    )

    rpn_head = compile_rpn(
        in_channels=in_channels,
        num_anchors=rpn_anchor_generator.num_anchors_per_location()[0]
    )

    box_roi_pool = compile_fast_roi(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio
    )

    mask_roi_pooler = compile_mask_roi(
        featmap_names=mask_featmap_names,
        output_size=mask_output_size,
        sampling_ratio=mask_sampling_ratio
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pooler,
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std
    )

    return model
