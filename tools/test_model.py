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

import unip
from unip.utils.evaluation import cal_flops

import utils

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    # cfg = get_detector_cfg(args.config)
    cfg = Config.fromfile(args.config)

    DefaultScope.get_instance("prune", scope_name="mmdet3d")
    model = MODELS.build(cfg.model)
    data = utils.get_example_data(
        model,
        num_gt_instance=cfg.p_num_gt_instance,
        points_feat_dim=cfg.p_points_feat_dim,
    )

    flops, params, clever_print = cal_flops(model, data)
    # print(model)
    print(f"Original: {clever_print}")

if __name__ == "__main__":
    main()
