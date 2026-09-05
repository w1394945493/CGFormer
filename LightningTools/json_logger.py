import json
import os
import time

import torch
from pytorch_lightning.callbacks import Callback


class JsonLogger(Callback):
    """Write training losses and evaluation metrics as JSON Lines."""

    def __init__(self, log_path, interval):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.batch_start = None
        self.batch_end = None
        self.data_time = 0.0
        self.grad_norm = None
        self.val_batch_start = None
        self.val_data_time = 0.0
        self.val_time = 0.0
        self.val_batches = 0

    def setup(self, trainer, pl_module, stage):
        # DDP child processes may enter the program independently. Use rank 0's
        # path everywhere so that a run always has a single JSON file.
        self.log_path = trainer.strategy.broadcast(self.log_path, src=0)

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu()) if value.numel() == 1 else None
        return float(value) if isinstance(value, (int, float)) else None

    def _write(self, trainer, record):
        if not trainer.is_global_zero:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        record = {
            key: float(f"{value:.6g}") if isinstance(value, float) else value
            for key, value in record.items()
        }
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self.data_time = 0.0 if self.batch_end is None else now - self.batch_end
        self.batch_start = now

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if (trainer.global_step + 1) % self.interval:
            return
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        self.grad_norm = float(torch.stack(norms).norm(2).cpu()) if norms else 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        step = trainer.global_step
        batch_time = now - self.batch_start
        self.batch_end = now
        if step == 0 or step % self.interval:
            return

        record = {}
        for total_key in ("train/loss", "train/loss_step"):
            total_loss = self._scalar(trainer.callback_metrics.get(total_key))
            if total_loss is not None:
                record["loss"] = total_loss
                break

        for key, value in trainer.callback_metrics.items():
            if not key.startswith("train/") or key.endswith("_epoch"):
                continue
            name = key.removeprefix("train/").removesuffix("_step")
            if "loss" not in name.lower() or name in record:
                continue
            scalar = self._scalar(value)
            if scalar is not None:
                record[name] = scalar

        lrs = [group["lr"] for group in trainer.optimizers[0].param_groups]
        record["base_lr"] = max(lrs)
        record["lr"] = min(lrs)
        record["data_time"] = self.data_time
        record["grad_norm"] = self.grad_norm
        record["time"] = batch_time
        record["epoch"] = trainer.current_epoch + 1
        record["iter"] = batch_idx + 1
        record["memory"] = int(
            torch.cuda.max_memory_allocated(pl_module.device) / 1024 ** 2
        )
        record["step"] = step
        self._write(trainer, record)

    def on_validation_start(self, trainer, pl_module):
        self.val_batch_start = None
        self.val_data_time = 0.0
        self.val_time = 0.0
        self.val_batches = 0

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        now = time.perf_counter()
        if self.val_batch_start is not None:
            self.val_data_time += now - self.val_batch_start
        self.val_batch_start = now

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_time += time.perf_counter() - self.val_batch_start
        self.val_batch_start = time.perf_counter()
        self.val_batches += 1

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._write_eval(trainer, pl_module, "val")

    def _write_eval(self, trainer, pl_module, stage):
        metrics = trainer.callback_metrics
        record = {}
        class_gt = {}
        for key, value in metrics.items():
            prefix = f"{stage}/class_gt/"
            if key.startswith(prefix):
                class_gt[key.removeprefix(prefix)] = self._scalar(value)

        semantic_ious = []
        for key, value in metrics.items():
            prefix = f"{stage}/class_iou/"
            if not key.startswith(prefix):
                continue
            name = key.removeprefix(prefix)
            scalar = self._scalar(value)
            if name != pl_module.class_names[0] and scalar is not None:
                semantic_ious.append(scalar)
            if class_gt.get(name, 0) <= 0:
                record[name] = None
            elif scalar is not None:
                record[name] = scalar * 100.0

        for source, target in (
            (f"{stage}/IoU", "iou"),
            (f"{stage}/Precision", "precision"),
            (f"{stage}/Recall", "recall"),
        ):
            scalar = self._scalar(metrics.get(source))
            if scalar is not None:
                record[target] = scalar * 100.0

        # Prefer the model's globally aggregated mIoU. The fallback keeps this
        # callback useful with models that only expose per-class values.
        miou = self._scalar(metrics.get(f"{stage}/mIoU"))
        if miou is not None:
            record["miou"] = miou * 100.0
        elif semantic_ious:
            record["miou"] = sum(semantic_ious) / len(semantic_ious) * 100.0

        count = max(self.val_batches, 1)
        record["data_time"] = self.val_data_time / count
        record["time"] = self.val_time / count
        record["epoch"] = trainer.current_epoch + 1
        record["step"] = trainer.global_step
        self._write(trainer, record)

    def on_test_start(self, trainer, pl_module):
        self.on_validation_start(trainer, pl_module)

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_test_end(self, trainer, pl_module):
        self._write_eval(trainer, pl_module, "test")
