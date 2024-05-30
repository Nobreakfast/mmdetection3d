# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine import DefaultScope

from mmdet3d.utils import replace_ceph_backend
from mmdet3d.testing import create_detector_inputs, get_detector_cfg, setup_seed

import unip
from unip.utils.evaluation import cal_flops
from unip.utils.data_type import DEVICE

import utils


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument("--load_prune_pt", help="load pruned model", default=None)
    parser.add_argument("--load_prune_pth", help="load pth model", default=None)
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options', # --cfg-options "test_evaluator.pklfile_prefix=./test"
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.load_prune_pt is None:
        print("[UNIP] No .pt file to load. Trying to prune the model (which may cause error in [Load .pth])")
        model = runner.model
        data = utils.get_example_data(model, num_gt_instance=cfg.p_num_gt_instance, points_feat_dim=cfg.p_points_feat_dim)

        pruner = unip.prune(
            cfg.p_pruner, model, data, verbose=cfg.p_verbose, **cfg.p_others
        )
        pruner.prune()
        # output = model(**data)
        model.zero_grad()
    else:
        print("[UNIP] Load .pt file from", args.load_prune_pt)
        runner.model = torch.load(args.load_prune_pt)
    if args.load_prune_pth is not None:
        print("[UNIP] Load .pth file from", args.load_pth)
        runner._load_from = args.load_pth
        runner.load_or_resume()
    else:
        print("[UNIP] No .pth file to load")

    data = utils.get_example_data(runner.model, num_gt_instance=2, points_feat_dim=7)
    flops, params, clever_print = cal_flops(runner.model, data, DEVICE)
    print(f"Pruned: {clever_print}")


    # start testing
    runner.test()

    # save model
    # save_path = f"work_dirs/{args.config.split('/')[-1].split('.')[0]}/pruned_last.pt"
    # os.system(f"mkdir -p {osp.dirname(save_path)}")
    # torch.save(runner.model, save_path)

if __name__ == '__main__':
    main()
