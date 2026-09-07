import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from .modules.utils import Voxelization
import spconv.pytorch as spconv

@HEADS.register_module()
class VoxelProposalLayer(BaseModule):
    def __init__(
        self,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[256, 256, 32],
        data_config=None,
        init_cfg=None,
        **kwargs
    ):
        super(VoxelProposalLayer, self).__init__(init_cfg)

        self.data_config = data_config
        self.init_cfg = init_cfg
        self.voxelize = Voxelization(
            point_cloud_range=point_cloud_range, 
            spatial_shape=np.array(input_dimensions))
        
        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid) # 未归一化的坐标

        self.input_dimensions = input_dimensions
        
    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    # 将深度图中每个像素的(u v d)反投影为三维点，依次撤销图像增强、应用相机内参逆变换、执行cam-to-ego外参变换和BEV数据增强，最终得到ego/lidar系下的三维点云
    def depth2lidar(self, image_grid, depth, cam_params): # image_grid:(1 2 384 1280) # cam_params: 相机坐标到自车坐标旋转，相机坐标到自车坐标平移，相机内参 图像增强旋转矩阵 图像增强平移 BEV数据增强变换矩阵
        b, _, h, w = depth.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params # rots:(1 1 3 3) trans:(1 1 3) intrins:(1 1 4 4) post_rots:(1 1 3 3) post_trans:(1 1 3) bda:(1 4 4)
        # 将像素坐标和深度拼成(u v d)
        points = torch.cat([image_grid.repeat(b, 1, 1, 1), depth], dim=1) # (1 3 384 1280) # [b, 3, h, w] 
        points = points.view(b, 3, h * w).permute(0, 2, 1) # (1 491520 3)
        # 撤销图像增强，原始图像通常经过resize, crop, flip操作，增强后的像素坐标与原始相机内参不再对应 p_aug = Rp_raw+t -> p_raw = R-1(p_aug-t)
        # undo pos-transformation
        points = points - post_trans.view(b, 1, 3) # (1 491520 3) # 先撤销平移
        points = torch.inverse(post_rots).view(b, 1, 3, 3).matmul(points.unsqueeze(-1)) # (1 491520 3 1) # 再与旋转矩阵逆相乘

        # cam to ego 
        points = torch.cat([points[:, :, 0:2, :] * points[:, :, 2:3, :], points[:, :, 2:3, :]], dim=2) # 将(u v d)变换成齐次坐标(ud,vd,d)的形式
        # 处理4x4投影矩阵的平移列
        if intrins.shape[3] == 4: # (ud vd d) =Kpc+s -> pc = K-1[(ud vd d)-s]
            shift = intrins[:, :, :3, 3] 
            points = points - shift.view(b, 1, 3, 1) # 先减去平移列
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins)) # 再乘以K的逆矩阵和Rc->e
        points = combine.view(b, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(b, 1, 3) # 再加入相机在ego坐标系中的平移，得到ego/lidar坐标
        # 应用bev数据增强：训练时三维空间执行的数据增强操作由bda表示
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1) # 先把三维点转换为齐次坐标(X Y Z)->(X Y Z 1)
            points = bda.view(b, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1) # (X' Y' Z' 1) = R(X Y Z 1)
            points = points[..., :3] # 最后去除齐次坐标
        else:
            points = bda.view(b, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points # (1 491520 3)
    
    def lidar2voxel(self, points, device):
        points_reshape = []
        batch_idx = []
        tensor = torch.ones((1,), dtype=torch.long).to(device)

        for i, pc in enumerate(points):
            points_reshape.append(pc)
            batch_idx.append(tensor.new_full((pc.shape[0],), i))
        
        points_reshape, batch_idx = torch.cat(points_reshape), torch.cat(batch_idx)

        unq, unq_inv = self.voxelize(points_reshape, batch_idx)

        return unq, unq_inv
    
    def forward(self, cam_params, img_metas):
        depth = img_metas['stereo_depth'] # (1 1 384 1280)
        points = self.depth2lidar(self.image_grid, depth, cam_params) # 
        unq, unq_inv = self.lidar2voxel(points, points.device) # (17618 4)
        sparse_tensor = spconv.SparseConvTensor(
            torch.ones(unq.shape[0], dtype=torch.float32).view(-1, 1).to(points.device),
            unq.int(), spatial_shape=self.input_dimensions, batch_size=(torch.max(unq[:, 0] + 1))
            )
        input = sparse_tensor.dense() # 把被深度点命中的稀疏体素坐标转换成稠密的二值体素网格

        return input