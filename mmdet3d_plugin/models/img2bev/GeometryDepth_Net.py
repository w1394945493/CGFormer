import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS
from mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from .modules.Mono_DepthNet_modules import DepthNet
from .modules.Stereo_Depth_Net_modules import SimpleUnet, convbn_2d, DepthAggregation
import pdb

class StereoVolumeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StereoVolumeEncoder, self).__init__()
        self.stem = convbn_2d(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.Unet = nn.Sequential(
            SimpleUnet(out_channels)
        )
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        x = self.Unet(x)
        x = self.conv_out(x)
        return x

@NECKS.register_module()
class GeometryDepth_Net(BaseModule):
    """预测上下文特征及几何增强的离散深度分布。

    深度不仅用于辅助监督，还同时服务于 CAQG 的 2D-to-3D lifting 和
    3D deformable cross-attention，是 CGFormer 消除射线深度歧义的基础。
    """
    def __init__(
        self,
        downsample=8,
        numC_input=512,
        numC_Trans=64,
        cam_channels=27,
        grid_config=None,
        loss_depth_weight=1.0,
        loss_depth_type='bce',
    ):
        super(GeometryDepth_Net, self).__init__()

        self.downsample = downsample
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config

        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config['dbound']
        self.stereo_volume_encoder = StereoVolumeEncoder(
            in_channels=D, out_channels=D
        )
        self.depth_net = DepthNet(self.numC_input, self.numC_input,
                                  self.numC_Trans, self.D, cam_channels=self.cam_channels)

        self.loss_depth_weight = loss_depth_weight
        self.loss_depth_type = loss_depth_type

        self.constant_std = 0.5

        self.depth_aggregation = DepthAggregation(embed_dims=32, out_channels=1)

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())

        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)

        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))

        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]

        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)

        return depth_loss

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)

        else:
            pdb.set_trace()

        return self.loss_depth_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths_vals, gt_depths.float()

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        """把相机参数与数据增强参数编码成 DepthNet 的条件向量。

        该函数不直接生成图像/深度特征，而是为 batch 中的每个相机整理
        一个低维 camera-aware 向量。DepthNet 随后用 MLP 和 SE 模块把这些
        参数映射为通道权重，使深度预测及 context feature 能适应不同的
        相机内外参和数据增强变换。

        Args:
            rot: 相机坐标系到自车坐标系的旋转，形状为 (B, N, 3, 3)。
            tran: 相机坐标系到自车坐标系的平移，形状为 (B, N, 3)。
            intrin: 相机内参/投影矩阵，KITTI 中通常为齐次 4x4 矩阵。
            post_rot: 图像 resize、crop、flip 后对应的旋转/缩放矩阵。
            post_tran: 图像增强产生的二维平移。
            bda: BEV Data Augmentation 变换；未提供时使用单位矩阵。

        Returns:
            mlp_input: 每个相机的条件向量，形状为 (B, N, C)。标准
                KITTI 配置使用齐次 4x4 bda，因此 C=33。
        """
        B, N, _, _ = rot.shape

        if bda is None:
            # 没有进行 BEV/体素空间增强时，以 3x3 单位变换作为默认值。
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)

        # 同一场景的 bda 对所有 N 个相机相同，这里扩展出相机维度，
        # 以便与形状为 (B, N, ...) 的其他相机参数逐项拼接。
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)

        if intrin.shape[-1] == 4: # (1 1 4 4)
            # KITTI 分支：从齐次投影矩阵提取 7 个有效参数，包括
            # fx、fy、cx、cy，以及与相机基线/投影平移有关的 3 项。
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],  # fx：水平方向焦距
                intrin[:, :, 1, 1],  # fy：垂直方向焦距
                intrin[:, :, 0, 2],  # cx：主点横坐标
                intrin[:, :, 1, 2],  # cy：主点纵坐标
                intrin[:, :, 0, 3],  # 投影矩阵 x 方向平移项
                intrin[:, :, 1, 3],  # 投影矩阵 y 方向平移项
                intrin[:, :, 2, 3],  # 投影矩阵 z 方向平移项
                post_rot[:, :, 0, 0],  # 图像增强的 2x2 线性部分
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],    # 图像增强 x 平移
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],    # 图像增强 y 平移
                bda[:, :, 0, 0],       # BEV 增强的 xy 线性部分
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],       # BEV 增强的 z 缩放
            ], dim=-1)

            if bda.shape[-1] == 4:
                # 齐次 4x4 bda 还包含 xyz 三个平移量：
                # 7（内参）+ 6（图像增强）+ 5（bda线性）+ 3（bda平移）=21。
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            # 普通 3x3 相机内参没有投影平移列，只提取 fx、fy、cx、cy；
            # 其余 6 个图像增强参数和 5 个 bda 参数与 KITTI 分支相同。
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)

        # 把 3x3 旋转和 3x1 平移组成 3x4 sensor-to-ego 外参，再展平为
        # 12 维。外参告诉 DepthNet 当前相机在自车坐标系中的姿态和位置。
        sensor2ego = torch.cat(
            [rot, tran.reshape(B, N, 3, 1)], dim=-1
        ).reshape(B, N, -1)  # (B, N, 12)

        # 标准 KITTI + 4x4 bda：21 维内参/增强信息 + 12 维外参 = 33 维。
        # 输出随后送入 DepthNet.bn、depth_mlp/context_mlp 和两个 SE 模块。
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)  # (B, N, 33)

        return mlp_input

    def forward(self, input, img_metas):
        """融合单目预测与几何深度先验，输出上下文特征和深度分布。"""
        # x 是图像编码器输出；其余参数描述相机内外参和图像/BEV增强。
        # mlp_input 已由 get_mlp_input() 整理为每个相机的条件向量。
        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = input
        # 读取外部几何/立体深度图，作为比纯单目预测更可靠的几何先验。
        stereo_depth = img_metas['stereo_depth']  # (B, N, H_img, W_img)

        # x: (批大小, 相机数, 特征通道, 特征图高, 特征图宽)。
        B, N, C, H, W = x.shape
        # DepthNet 是普通 2D 网络，因此把 B、N 合并后逐张图像处理。
        x = x.view(B * N, C, H, W)  # 合并批次和相机维

        # 单目分支：DepthNet 使用 mlp_input 通过 MLP/SE 调制图像特征，
        # 使输出能够感知焦距、相机位姿和数据增强带来的几何变化。
        x = self.depth_net(x, mlp_input)  # (B*N, D+numC_Trans, H, W)
        # 前 D 个通道表示每个像素属于各离散深度区间的未归一化分数。
        mono_digit = x[:, :self.D, ...]  # (B*N, D, H, W)
        # 沿 D 个深度 bin 做 softmax，得到每个像素的单目深度概率分布。
        mono_volume = self.get_depth_dist(mono_digit)  # (B*N, D, H, W)
        # 后 numC_Trans 个通道保留图像语义，后续用于 CAQG 和 CGVT。
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]  # (B*N, C_ctx, H, W)

        # 几何分支：将连续深度图降采样到特征图尺度，并量化到与单目
        # 分支相同的 D 个深度 bin，方便两种深度信息逐位置融合。
        _, stereo_volume = self.get_downsampled_gt_depth(stereo_depth)  # one-hot 深度
        # 上一步返回展平的 (B*N*H*W,D)，这里恢复为 CNN 所需的
        # (B,D,H,W)。当前写法将 N 并入通道，因此标准配置要求 N=1。
        stereo_volume = stereo_volume.view(B, H, W, -1).permute(0, 3, 1, 2)
        # 通过卷积和 U-Net 聚合邻域信息，补充并平滑稀疏的几何深度线索。
        stereo_volume = self.stereo_volume_encoder(stereo_volume)  # (B, D, H, W)
        stereo_volume = self.get_depth_dist(stereo_volume)  # 几何深度概率分布

        # 融合分支：先进行“单目关注几何、几何关注单目”的双向邻域注意力，
        # 再由 3D U-Net 同时建模深度轴和图像空间，输出融合深度 logits。
        depth_volume = self.depth_aggregation(stereo_volume, mono_volume)
        # 最后沿深度维归一化。该分布既用于把 img_feat 提升到 3D，
        # 也用于约束 3D deformable cross-attention 的深度采样。
        depth_volume = self.get_depth_dist(depth_volume)  # (B, D, H, W)

        # 恢复 context feature 的相机维；深度输出按当前单相机实现保持 4 维。
        return img_feat.view(B, N, -1, H, W), depth_volume
