import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class CGFormer(BaseModule):
    """CGFormer 主干流程。

    论文中的三项关键设计在这里串联起来：先预测图像上下文与深度，
    再用二者生成与当前场景相关的体素 query，并通过带深度约束的
    CGVT 将 2D 特征提升到 3D；最后由局部 Voxel 与全局 TPV 分支
    共同增强体素特征，输出语义占据结果。
    """
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None
        ):
        super().__init__()

        # ① 2D 图像编码器：EfficientNet Backbone 提取多尺度图像特征，
        # FPN Neck 将不同尺度特征对齐并融合，作为后续深度预测和
        # 2D-to-3D 特征提升的视觉输入。
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        # ② Geometry Depth Net：由图像特征同时预测 context feature 和
        # 离散深度分布。context 负责携带语义信息，depth 负责提供几何约束；
        # 二者会共同用于 CAQG，并将 depth 继续传给 3D deformable attention。
        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            # ③ Context-Aware Query Generator（CAQG）的视图变换部分：
            # 按深度概率把当前图像的 context feature 从视锥体聚合到体素空间，
            # 生成与输入场景相关的粗 3D volume，用来初始化 voxel queries。
            self.img_view_transformer = builder.build_neck(img_view_transformer)

        # ④ 稀疏体素提议：根据深度/投影关系选择当前相机可见的候选体素。
        # 后续只对这些体素执行较昂贵的 cross-attention，以降低计算和显存。
        self.proposal_layer = builder.build_head(proposal_layer)

        # ⑤ Context and Geometry Aware Voxel Transformer（CGVT）：
        # - 将 CAQG 的粗 3D volume 注入固定 voxel embedding，形成场景相关 query；
        # - 用深度引导的 3D deformable cross-attention 聚合可见区域图像特征；
        # - 用 deformable self-attention 将信息扩散到遮挡和不可见体素。
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            # ⑥ Local-and-Global Encoder（LGE）主体，配置中对应 Fuser：
            # 局部 Voxel 分支保留细粒度 3D 几何，TPV 分支在 XY/YZ/ZX
            # 三个平面建模全局上下文，最后为每个体素动态加权融合四路特征。
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            # 可选的 3D neck，用于进一步进行多尺度体素特征融合；当前标准
            # CGFormer 配置未单独启用，主要增强过程由上面的 LGE 完成。
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)

        # ⑦ SSC 解码头：把增强后的 3D voxel feature 上采样到目标占据分辨率，
        # 为每个体素预测 empty/各语义类别，并在训练阶段计算 SSC 损失。
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # 是否额外使用深度监督。标准完整配置默认关闭，主要依赖预训练好的
        # Geometry Depth Net；设置为 True 时会把 depth loss 加入总训练损失。
        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        # 图像编码器提供多尺度视觉语义，GeometryDepthNet 将其拆分为
        # context feature 与离散深度分布；后两者分别承载语义和几何信息。
        img_enc_feats = self.image_encoder(img_inputs[0]) # (1 1 640 48 160)  # img_inputs[0]:(1 1 3 384 1280)

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7]) # (1 1 33)
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas) # (1 1 128 48 160) (1 112 48 160) # 上下文特征与离散深度分布
        
        if hasattr(self, 'img_view_transformer'):
            # Context-Aware Query Generator（CAQG）：利用预测深度将当前
            # 图像的 context feature 提升并聚合到体素空间。该粗体素特征
            # 会初始化 query，避免所有场景都只使用相同的可学习 query。
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        # 仅让可见/候选体素参与昂贵的 cross-attention，随后再通过
        # self-attention 将信息由可见区域扩散到遮挡及不可见区域。
        proposal = self.proposal_layer(img_inputs[1:7], img_metas)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=img_inputs[1:7],
            lss_volume=coarse_queries,
            img_metas=img_metas,
            # CGVT 的 3D deformable cross-attention 使用该深度分布区分
            # 投影到相近二维像素、但处于不同深度位置的体素。
            mlvl_dpt_dists=[depth.unsqueeze(1)]
        )

        return x, depth
    
    def occ_encoder(self, x):
        # 配置中的 Fuser 即 Local-and-Global Encoder（LGE）：局部 Voxel
        # 分支保留细粒度几何，全局 TPV 分支建模三个平面上的长程依赖。
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred, # (1 256 256 32)
            'gt_occ': gt_occ # (1 256 256 32)
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
