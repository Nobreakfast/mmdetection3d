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
from thop import profile, clever_format


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument('checkpoint', help='checkpoint file', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)

    DefaultScope.get_instance("prune", scope_name="mmdet3d")
    model = MODELS.build(cfg.model)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    # type 1: directly use the modules
    ignore_modules = {
        model.middle_encoder: None,
        model.voxel_encoder: None,
    }
    # type 2: use the types and our function
    # ignore_types = {
    #     PointPillarsScatter: None,
    # }
    num_gt_instance = 2
    packed_inputs = create_detector_inputs(
        num_gt_instance=num_gt_instance, points_feat_dim=7
    )

    data = model.data_preprocessor(packed_inputs, True)

    flops, params, clever_print = cal_flops(model, data, DEVICE)
    print(f"Original: {clever_print}")
    # print(profile(model, inputs=[data["inputs"], data["data_samples"]]))

    pruner = unip.prune(
        "OneShot", model, data, ratio=0.4, verbose=False, ignore_modules=ignore_modules
    )
    # pruner.plot(group=False, save_path=f"work_dirs/fig/{time.time()}")
    pruner.prune()
    flops, params, clever_print = cal_flops(model, data, DEVICE)
    print(f"Pruned: {clever_print}")
    # output = model(**data)

    # save model
    save_path = f"work_dirs/{args.config.split('/')[-1].split('.')[0]}/pruned.pt"
    os.system(f"mkdir -p {osp.dirname(save_path)}")
    torch.save(model, save_path)

    # load model
    model = torch.load(save_path)

    output = model(**data)
    for o in output:
        if isinstance(o, torch.Tensor):
            print(o.shape)
            continue
        for sub_o in o:
            print(sub_o.shape)


if __name__ == "__main__":
    main()
