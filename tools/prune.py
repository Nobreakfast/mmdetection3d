# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import torch

from mmengine import DefaultScope

from mmdet3d.registry import MODELS
import mmdet3d.models

from mmdet3d.utils import replace_ceph_backend
from mmdet3d.testing import create_detector_inputs, get_detector_cfg, setup_seed
from mmdet3d.models.middle_encoders import PointPillarsScatter

import unip
from unip.utils.evaluation import cal_flops
from unip.utils.data_type import DEVICE

import utils


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    DefaultScope.get_instance("prune", scope_name="mmdet3d")

    if args.checkpoint is not None:
        print("[UNIP] Load checkpoint from", args.checkpoint)
        cfg.model.load_from = args.checkpoint
    model = MODELS.build(cfg.model)

    data = utils.get_example_data(
        model,
        num_gt_instance=cfg.p_num_gt_instance,
        points_feat_dim=cfg.p_points_feat_dim,
    )

    flops, params, clever_print = cal_flops(model, data, DEVICE)
    print(f"Original: {clever_print}")

    pruner = unip.prune(
        cfg.p_pruner, model, data, verbose=cfg.p_verbose, **cfg.p_others
    )
    # pruner.plot(group=False, save_path=f"work_dirs/fig/{time.time()}")
    pruner.prune()

    flops, params, clever_print = cal_flops(model, data, DEVICE)
    print(f"Pruned: {clever_print}")

    # save model
    save_path = cfg.work_dir + "pruned_model.pt"
    os.system(f"mkdir -p {osp.dirname(save_path)}")
    torch.save(model, save_path)


if __name__ == "__main__":
    main()
