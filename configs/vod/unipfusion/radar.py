"""
global settings
"""

default_scope = "mmdet3d"
backend_args = None
custom_imports = dict(
    imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False
)

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
batch_size = 2
num_workers = 4
data_root = "data/vod5f/"
work_dir = "work_dirs/unipfusion_radar/train/"
submission_prefix = work_dir + "results/"
pklfile_prefix = work_dir + "pkl/"
optim_type = "AdamW"

# dataset settings
dataset_type = "KittiDataset"
input_modality = dict(use_lidar=True, use_camera=False)
class_names = ["Pedestrian", "Cyclist", "Car"]
metainfo = dict(classes=class_names)
voxel_size = [0.16, 0.16, 5]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
output_shape = [320, 320]
image_size = [1216, 1936]

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
lr = 0.0001
epoch_num = 24

# log settings
log_level = "INFO"
load_from = None
resume = False
checkpoint_num = 1

# model
model = dict(
    type="UniPFusion",
    modality=input_modality,
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=max_num_points,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
        ),
    ),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=7,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter", in_channels=64, output_shape=output_shape
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
    ),
    # fusion_layer=dict(type="ConvFuser", in_channels=[64, 64], out_channels=64),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=anchor_ranges,
            sizes=anchor_sizes,
            rotations=[0, 1.57],
            reshape_out=False,
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
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Cyclist
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Car
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
        ],
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
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=7,
        use_dim=7,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_labels_3d", "gt_bboxes_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=7,
        use_dim=7,
        backend_args=backend_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["points", "gt_labels_3d", "gt_bboxes_3d"]),
]
eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=7,
        use_dim=7,
        backend_args=backend_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["points", "gt_labels_3d", "gt_bboxes_3d"]),
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
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, betas=(0.95, 0.99), weight_decay=1e-4),
    clip_grad=dict(max_norm=5, norm_type=2),
)
param_scheduler = [
    # dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        T_max=24,
        end=24,
        by_epoch=True,
        eta_min=lr / 10,
    ),
]
auto_scale_lr = dict(enable=True, base_batch_size=8)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()
