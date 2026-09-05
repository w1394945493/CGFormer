import os
from datetime import datetime
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from LightningTools.json_logger import JsonLogger
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.profiler import SimpleProfiler # 旧版写法
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def is_process_rank_zero():
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))) == 0


def get_shared_timestamp():
    key = 'CGFORMER_RUN_TIMESTAMP'
    if key not in os.environ:
        os.environ[key] = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.environ[key]

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    timestamp = get_shared_timestamp()
    run_dir = os.path.join(log_folder, timestamp)
    if is_process_rank_zero():
        misc.check_path(run_dir)
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name=timestamp,
        version='vis_data'
    )
    json_logger = JsonLogger(
        os.path.join(run_dir, '{}.json'.format(timestamp)),
        config['log_every_n_steps']
    )

    if is_process_rank_zero():
        config.dump(os.path.join(run_dir, 'config.py'))
    profiler = SimpleProfiler(dirpath=run_dir, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_folder,
        monitor='val/mIoU',
        mode='max',
        save_last=False,
        filename='best',
        enable_version_counter=False)

    # Refresh one recoverable epoch_N checkpoint after every completed epoch.
    # It lives outside timestamped log directories so a new run can find it.
    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=log_folder,
        monitor=None,
        save_last=False,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        filename='epoch_{epoch:.0f}',
        enable_version_counter=False)
    
    if not config.eval:
        # An explicit checkpoint takes precedence. Otherwise resume from the
        # largest epoch_N.ckpt in log_folder.
        resume_ckpt_path = config['ckpt_path']
        if resume_ckpt_path is None:
            epoch_ckpts = []
            for filename in os.listdir(log_folder):
                if not filename.startswith('epoch_') or not filename.endswith('.ckpt'):
                    continue
                epoch_text = filename[len('epoch_'):-len('.ckpt')]
                if epoch_text.isdigit():
                    epoch_ckpts.append((int(epoch_text), filename))

            if epoch_ckpts:
                _, latest_filename = max(epoch_ckpts, key=lambda item: item[0])
                resume_ckpt_path = os.path.join(log_folder, latest_filename)
                print('发现训练断点，将从以下位置继续训练：{}'.format(
                    resume_ckpt_path))
            else:
                print('未发现epoch_N.ckpt训练断点，将从头开始训练：{}'.format(
                    log_folder))
        else:
            if not os.path.isfile(resume_ckpt_path):
                raise FileNotFoundError(
                    '指定的训练断点不存在：{}'.format(resume_ckpt_path))
            print('使用指定的训练断点继续训练：{}'.format(
                resume_ckpt_path))

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                find_unused_parameters=False
            ),
            max_steps=config.training_steps,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_epoch,
                LearningRateMonitor(logging_interval='step'),
                json_logger
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch']
        )
        trainer.fit(
            model=model,
            datamodule=data_dm,
            ckpt_path=resume_ckpt_path)
    else:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                find_unused_parameters=False
            ),
            callbacks=[json_logger],
            logger=tb_logger,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])
