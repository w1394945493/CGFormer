from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class LocalAggregator(BaseModule):
    """LGE 的局部分支：直接在 3D voxel 上保留细粒度几何与语义。"""
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
    ):
        super().__init__()
        self.local_encoder_backbone = builder.build_backbone(local_encoder_backbone)
        self.local_encoder_neck = builder.build_neck(local_encoder_neck)
    
    def forward(self, x):
        # 3D CNN/FPN 聚合邻域体素。该分支空间细节充分，与擅长全局
        # 建模但经过轴向压缩的 TPV 分支形成互补。
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)
        output = output[0]

        return output
