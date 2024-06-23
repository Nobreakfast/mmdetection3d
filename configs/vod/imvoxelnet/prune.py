"""
global settings
"""

default_scope = "mmdet3d"
backend_args = None
# custom_imports = dict(
#     imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False
# )

# prune settings
p_pruner = "OneShot"
p_verbose = False
p_others = {
    "grapher": "backward",
    "grouper": "add",
    "algorithm": "uniform",
    "score": "l1",
    "ratio": 0.5,
    "ignore_modules": {
        "middle_encoder": None,
        "voxel_encoder": None,
    },
}
p_num_gt_instance = 2
p_points_feat_dim = 7

# Important settings
batch_size = 4
num_workers = 4
data_root = "data/vod5f/"
work_dir = "work_dirs/vod_imvoxelnet_r5/prune/"
submission_prefix = work_dir + "results/"
pklfile_prefix = work_dir + "pkl/"
optim_type = "AdamW"

# dataset settings
dataset_type = "KittiDataset"
input_modality = dict(use_lidar=True, use_camera=True)
class_names = ["Pedestrian", "Cyclist", "Car"]
metainfo = dict(classes=class_names)
voxel_size = [0.16, 0.16, 5]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
output_shape = [320, 320]
image_size = [1216, 1936]
# image_size = [608, 968]

# model settings
max_num_points = 10
anchor_ranges = [
    [0, -25.6, -0.6, 51.2, 25.6, -0.6],
    [0, -25.6, -0.6, 51.2, 25.6, -0.6],
    [0, -25.6, -1.78, 51.2, 25.6, -1.78],
]
anchor_sizes = [
    [0.8, 0.6, 1.73],
    [1.76, 0.6, 1.73],
    [3.9, 1.6, 1.56],
]

# training settings
lr = 0.001
epoch_num = 80

# log settings
log_level = "INFO"
load_from = None
resume = False
checkpoint_num = 1

# model
model = dict(
    type="ImVoxelNet",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        style="pytorch",
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4,
    ),
    neck_3d=dict(type="OutdoorImVoxelNeck", in_channels=64, out_channels=256),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[-0.16, -39.68, -1.78, 68.96, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2
        ),
    ),
    n_voxels=[216, 248, 12],
    coord_type="LIDAR",
    prior_generator=dict(
        type="AlignedAnchor3DRangeGenerator",
        ranges=[[-0.16, -39.68, -3.08, 68.96, 39.68, 0.76]],
        rotations=[0.0],
    ),
    train_cfg=dict(
        assigner=dict(
            type="Max3DIoUAssigner",
            iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=checkpoint_num),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

# pipeline
train_pipeline = [
    dict(type="LoadAnnotations3D", backend_args=backend_args),
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="RandomResize", scale=[(1173, 352), (1387, 416)], keep_ratio=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["img", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="Resize", scale=(1280, 384), keep_ratio=True),
    dict(type="Pack3DDetInputs", keys=["img"]),
]
eval_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="Resize", scale=(1280, 384), keep_ratio=True),
    dict(type="Pack3DDetInputs", keys=["img"]),
]

# datasets
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="kitti_infos_train.pkl",
            data_prefix=dict(pts="training/velodyne", img="training/image_2"),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts="training/velodyne", img="training/image_2"),
        ann_file="kitti_infos_val.pkl",
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts="training/velodyne", img="training/image_2"),
        ann_file="kitti_infos_val.pkl",
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_evaluator = dict(
    type="KittiMetric",
    ann_file=data_root + "kitti_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)

test_evaluator = dict(
    type="KittiMetric",
    ann_file=data_root + "kitti_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
    format_only=True,
    submission_prefix=submission_prefix,
    pklfile_prefix=pklfile_prefix,
)
# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type=optim_type, lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True,
    ),
]
auto_scale_lr = dict(enable=True, base_batch_size=4)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()
