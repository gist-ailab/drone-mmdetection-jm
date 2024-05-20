# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo
import mcdet
import easydict

CUDA_VISIBLE_DEVICES = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


def parse_args():
    # parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('config', help='train config file path', default= '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas-rgbt-v1.py')
    # parser.add_argument('--work-dir', help='the dir to save logs and models', default = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/')
    # parser.add_argument('--amp', action='store_true', default=False, help='enable automatic-mixed-precision training')
    # parser.add_argument('--auto-scale-lr', action='store_true', default=True, help='enable automatically scaling LR.')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    # args = parser.parse_args()
    args = easydict.EasyDict({
        'config': '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas-rgbt-v1.py',
        'work_dir': '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/Debug',
        'amp': False,
        'auto_scale_lr': True,
        'launcher': 'none',
        'local_rank': 0,
        'resume': None
    })


    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
