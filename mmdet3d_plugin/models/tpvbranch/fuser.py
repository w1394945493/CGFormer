from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class Fuser(BaseModule):
    """动态融合局部 voxel 与三个全局 TPV 表示。

    不采用固定相加，而是为每个体素预测四个归一化权重，使网络能按
    场景位置选择局部几何或某个方向上的全局上下文。
    """
    def __init__(
        self,
        embed_dims=128,
        global_aggregator=None,
        local_aggregator=None
    ):
        super().__init__()
        self.global_aggregator = builder.build_backbone(global_aggregator)
        self.local_aggregator = builder.build_backbone(local_aggregator)

        self.combine_coeff = nn.Sequential(
            nn.Conv3d(embed_dims, 4, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        local_feats = self.local_aggregator(x)
        global_feats = self.global_aggregator(x)

        # 四个 softmax 权重依次对应 local、XY、YZ、ZX；权重和为 1。
        weights = self.combine_coeff(local_feats)

        out_feats = local_feats * weights[:, 0:1, ...] + global_feats[0] * weights[:, 1:2, ...] + \
            global_feats[1] * weights[:, 2:3, ...] + global_feats[2] * weights[:, 3:4, ...]

        return out_feats
