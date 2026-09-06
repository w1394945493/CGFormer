# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule
from mmdet3d.ops.bev_pool import bev_pool
# from packages.mmdetection3d.mmdet3d.ops.bev_pool import bev_pool

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@NECKS.register_module()
class LSSViewTransformer(BaseModule):
    """将图像上下文按深度概率提升到 3D，生成场景相关的粗体素 query。

    这是论文 Context-Aware Query Generator 的核心 lifting/splatting
    部分：不同输入图像会产生不同的体素先验，而非共享同一组固定 query。
    """
    def __init__(
        self, grid_config=None, data_config=None, downsample=8,
    ):
        super().__init__()

        self.grid_config = grid_config
        self.data_config = data_config
        self.downsample = downsample
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def voxel_pooling(self, geom_feats, x):
        # x 是视锥特征：(B,N,D,H,W,C)；geom_feats 是各特征对应的 ego 三维坐标：(B,N,D,H,W,3)。
        B, N, D, H, W, C = x.shape  # 依次取批大小、相机数、深度 bin 数、特征图高宽和通道数。
        Nprime = B * N * D * H * W  # 统计当前 batch 中全部“相机-深度-像素”采样点的数量。
        x = x.reshape(Nprime, C)  # 展平空间相关维度，使每一行对应一个三维采样点的 C 维特征。

        # bx 是第一个体素中心，bx-dx/2 是网格最小边界；除以 dx 将米制坐标量化为体素索引。
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()  # (x,y,z) -> (ix,iy,iz)。
        geom_feats = geom_feats.view(Nprime, 3)  # 将体素坐标展平，使其与上面的每一行特征一一对应。
        # 为每个采样点补充其所属 batch 的编号，形状为 (Nprime,1)。
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # 得到 (ix,iy,iz,batch_id)，供 bev_pool 分组聚合。

        # 只保留落在体素网格 [0,nx)×[0,ny)×[0,nz) 内的采样点，丢弃场景范围外的点。
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]  # 同步过滤视锥特征，仅保留有效三维点对应的特征。
        geom_feats = geom_feats[kept]  # 同步过滤体素索引，保持坐标与特征一一对应。

        # 将落入同一 (batch,ix,iy,iz) 体素的所有特征求和；算子输出顺序为 (B,C,Z,X,Y)。
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)  # 调整为后续网络使用的 (B,C,X,Y,Z)。

        return final  # 返回由图像视锥特征聚合得到的稠密三维体素特征。

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        """将视锥网格中的 (u, v, depth) 反投影为 ego 坐标系下的三维点。

        返回形状为 B x N x D x H/downsample x W/downsample x 3。
        """
        B, N, _ = trans.shape  # B 为批大小，N 为相机数量；trans 保存 camera-to-ego 平移。

        # self.frustum 中每个点为增强后图像上的 (u, v, depth)，形状为 (D,H,W,3)。
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # 撤销 resize/crop 等增强引入的平移。
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # 得到原始图像坐标 (u,v,d)，形状为 (B,N,D,H,W,3,1)。

        # 将 (u,v,d) 写成齐次像素射线上的深度点 (u*d,v*d,d)，以便随后乘 K^{-1}。
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],points[:, :, :, :, :, 2:3]), 5)  # 仍为 (B,N,D,H,W,3,1)。

        if intrins.shape[3] == 4:  # KITTI 的投影矩阵为 3x4，而一般相机内参为 3x3。
            shift = intrins[:, :, :3, 3]  # 取 3x4 投影矩阵最后一列（双目基线等引入的偏移项）。
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)  # 先移除 P=K[R|t] 中的偏移项。
            intrins = intrins[:, :, :3, :3]  # 保留左侧 3x3 内参矩阵 K。

        combine = rots.matmul(torch.inverse(intrins))  # 合并 K^{-1} 反投影与 camera-to-ego 旋转 R。
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # 得到旋转后的 ego 三维坐标。
        points += trans.view(B, N, 1, 1, 1, 3)  # 加 camera-to-ego 平移 t，完成 p_ego=R K^{-1}p+t。

        if bda.shape[-1] == 4:  # 若 BEV 数据增强矩阵是 4x4，则使用齐次坐标进行旋转、缩放和平移。
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)  # (x,y,z)->(x,y,z,1)。
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)  # 应用 BDA 变换。
            points = points[..., :3]  # 去掉齐次坐标，恢复 (x,y,z)。
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)  # 3x3 BDA 仅含旋转/缩放。

        return points  # (B,N,D,H/downsample,W/downsample,3)，每个深度 bin 对应一个 ego 三维点。


    def forward(self, feat, depth_prob, cam_params):
        B, N, C, H, W = feat.shape # (1 1 128 48 160)
        rots, trans, intrins, post_rots, post_trans, bda = cam_params

        if len(depth_prob.shape) == 4:
            db, cb, dh, dw = depth_prob.shape # (1 112 48 160)
            assert db == B * N
            depth_prob = depth_prob.view(B, N, cb, dh, dw) # (1 1 112 48 160)
        # *============================================*
        # * Lift & Splat：把每个像素的 context feature 按深度概率提升到 3D，生成场景相关的粗体素 query。
        # Lift：把每个像素的 context feature 与其各深度 bin 的概率相乘，
        # 沿相机射线展开成带语义权重的视锥体特征 (B,N,D,H,W,C)。
        volume = depth_prob.unsqueeze(2) * feat.unsqueeze(3) # (1 1 128 112 48 160)
        volume = volume.view(B, N, -1, self.D, H, W) # (1 1 128 112 48 160)
        volume = volume.permute(0, 1, 3, 4, 5, 2) # (1 1 112 48 160 128)

        # * Splat：根据相机内外参把视锥体采样点变换到自车坐标系，并将落在同一网格内的特征池化，得到用于初始化 Transformer query的 context-dependent 3D volume。
        # geom：锥体中每视个“像素 × 深度 bin”对应的三维坐标，坐标已经从图像/相机坐标系转换到了经过 BDA 增强的 ego（自车）坐标系。
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda) # (1 1 112 48 160 3)
        # 将视锥特征 -> 连续三维坐标量化 -> 过滤越界点 -> 同体素特征求和 -> 稠密三维体素特征 (B,C,X,Y,Z)。
        bev_feat = self.voxel_pooling(geom, volume) # (1 128 128 128 16)
        return bev_feat # (1 128 128 128 16)
