import os
import torch
import numpy as np
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from .metric import SSCMetrics
from mmdet3d.models import build_model
from .utils import get_inv_map
from mmcv.runner.checkpoint import load_checkpoint


class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        if 'load_from' in config:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])
        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']
    
    def forward(self, data_dict):
        return self.model(data_dict)

    def _get_global_stats(self, metric):
        """Aggregate metric counters across ranks before computing ratios."""
        counts = torch.as_tensor(
            np.concatenate((
                np.asarray([
                    metric.completion_tp,
                    metric.completion_fp,
                    metric.completion_fn,
                ], dtype=np.float64),
                metric.tps,
                metric.fps,
                metric.fns,
            )),
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

        num_classes = metric.n_classes
        completion_tp, completion_fp, completion_fn = counts[:3]
        tps = counts[3:3 + num_classes]
        fps = counts[3 + num_classes:3 + 2 * num_classes]
        fns = counts[3 + 2 * num_classes:]

        precision = completion_tp / (completion_tp + completion_fp).clamp_min(1e-5)
        recall = completion_tp / (completion_tp + completion_fn).clamp_min(1e-5)
        iou = completion_tp / (
            completion_tp + completion_fp + completion_fn
        ).clamp_min(1e-5)
        iou_ssc = tps / (tps + fps + fns).clamp_min(1e-5)

        return {
            'precision': precision.float(),
            'recall': recall.float(),
            'iou': iou.float(),
            'iou_ssc': iou_ssc.float(),
            'iou_ssc_mean': iou_ssc[1:].mean().float(),
            'class_gt': tps + fns,
        }
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
            
            self.train_metrics.add_batch(pred, gt_occ)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.val_metrics.add_batch(pred, gt_occ)
    
    def on_validation_epoch_end(self):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = self._get_global_stats(metric)

            if prefix == 'val':
                for name, iou in zip(self.class_names, stats['iou_ssc']):
                    self.log(
                        '{}/class_iou/{}'.format(prefix, name),
                        iou,
                        sync_dist=False)
                for name, count in zip(self.class_names, stats['class_gt']):
                    self.log(
                        '{}/class_gt/{}'.format(prefix, name),
                        count,
                        sync_dist=False)

            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=False)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=False)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=False)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=False)
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        gt_occ = output_dict['gt_occ']
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                print('\n save to {}'.format(save_file))
            
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
    
    def on_test_epoch_end(self):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = self._get_global_stats(metric)

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                if self.global_rank == 0:
                    print(name + ":", float(iou))
                self.log(
                    "{}/class_iou/{}".format(prefix, name),
                    iou,
                    sync_dist=False)
            for name, count in zip(self.class_names, stats['class_gt']):
                self.log(
                    "{}/class_gt/{}".format(prefix, name),
                    count,
                    sync_dist=False)

            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=False)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=False)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=False)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=False)
            metric.reset()
