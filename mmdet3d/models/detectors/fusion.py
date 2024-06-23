# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist

import numpy as np

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.utils import is_list_of
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector

import matplotlib.pyplot as plt


@MODELS.register_module()
class FusionDetector(Base3DDetector):
    """FusionDetector.

    This class serves as a base class for single-stage 3D detectors which
    directly and densely predict 3D bounding boxes on the output features
    of the backbone+neck.


    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        pts_voxel_encoder: OptConfigType = None,
        pts_middle_encoder: OptConfigType = None,
        img_backbone: OptConfigType = None,
        img_neck: OptConfigType = None,
        pts_backbone: OptConfigType = None,
        fusion_layer: OptConfigType = None,
        view_transform: OptConfigType = None,
        pts_neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        modality: OptConfigType = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.modality = modality
        if modality["use_lidar"]:
            self.pts_voxel_encoder = (
                MODELS.build(pts_voxel_encoder) if pts_voxel_encoder else None
            )
            self.pts_middle_encoder = (
                MODELS.build(pts_middle_encoder) if pts_middle_encoder else None
            )
        if modality["use_camera"]:
            self.img_backbone = MODELS.build(img_backbone) if img_backbone else None
            self.img_neck = MODELS.build(img_neck) if img_neck else None
            self.view_transform = (
                MODELS.build(view_transform) if view_transform else None
            )
        else:
            self.img_backbone = None
            self.img_neck = None
            self.view_transform = None
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer else None
        self.pts_backbone = MODELS.build(pts_backbone) if pts_backbone else None
        self.pts_neck = MODELS.build(pts_neck) if pts_neck else None
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()
        self.cam2img = np.asarray(
            [
                1495.468642,
                0.0,
                961.272442,
                0.0,
                0.0,
                1495.468642,
                624.89592,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape(4, 4)
        self.lidar2cam = np.asarray(
            [
                # -0.013857 -0.9997468 0.01772762 0.05283124 0.10934269 -0.01913807 -0.99381983 0.98100483 0.99390751 -0.01183297 0.1095802 1.44445002
                -0.013857,
                -0.9997468,
                0.01772762,
                0.05283124,
                0.10934269,
                -0.01913807,
                -0.99381983,
                0.98100483,
                0.99390751,
                -0.01183297,
                0.1095802,
                1.44445002,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape(4, 4)
        self.lidar2img = self.cam2img @ self.lidar2cam
        self.cam2lidar = np.linalg.inv(self.lidar2cam)

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def loss(
        self, batch_inputs_dict: dict, batch_data_samples: SampleList, **kwargs
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = dict()
        bbox_loss = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        losses.update(bbox_loss)
        return losses

    def predict(
        self, batch_inputs_dict: dict, batch_data_samples: SampleList, **kwargs
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples, results_list)
        return predictions

    def _forward(
        self, batch_inputs_dict: dict, data_samples: OptSampleList = None, **kwargs
    ) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict, data_samples)
        results = self.bbox_head.forward(x)
        return results

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor], data_samples: Dict[str, Tensor]
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        imgs = batch_inputs_dict["imgs"] if self.modality["use_camera"] else None
        points = batch_inputs_dict["points"]
        features = []
        if imgs is not None:
            imgs = imgs.unsqueeze(1).contiguous()
            lidar2img, cam2img, cam2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for data_sample in data_samples:
                lidar2img.append(data_sample.lidar2img)
                # lidar2img.append(self.lidar2img)
                cam2img.append(data_sample.cam2img)
                cam2lidar.append(np.linalg.inv(data_sample.lidar2cam))
                # cam2lidar.append(self.cam2lidar)
                img_aug_matrix.append(np.eye(4))
                lidar_aug_matrix.append(np.eye(4))

            lidar2img = imgs.new_tensor(np.asarray(lidar2img))
            lidar2img = lidar2img.unsqueeze(1).contiguous()
            cam2img = imgs.new_tensor(np.asarray(cam2img))
            cam2img = cam2img.unsqueeze(1).contiguous()
            cam2lidar = imgs.new_tensor(np.asarray(cam2lidar))
            cam2lidar = cam2lidar.unsqueeze(1).contiguous()
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feat = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2img,
                cam2img,
                cam2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                None,
            )
            img_feat = img_feat.transpose(-1, -2)
            features.append(img_feat)
        # if points is not None:
        if self.modality["use_lidar"]:
            pts_feature = self.extract_pts_feat(batch_inputs_dict)
            features.append(pts_feature)
        #import os

        #os.makedirs("/home/allen/Downloads/vis/", exist_ok=True)
        #plt.figure()
        #img_viz = imgs[0][0].detach().cpu().numpy().transpose(1, 2, 0)
        #plt.imshow(img_viz / 255)
        #plt.savefig("/home/allen/Downloads/vis/img.png")
        #plt.close()

        #plt.figure()
        #plt.imshow(features[0][0].sum(0).detach().cpu().numpy())
        #plt.savefig("/home/allen/Downloads/vis/img_feat.png")
        #plt.close()

        #plt.figure()
        #plt.imshow(features[1][0].sum(0).detach().cpu().numpy())
        #plt.savefig("/home/allen/Downloads/vis/pts_feat.png")
        #plt.close()
        #print(heel)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            # assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        voxel_dict = batch_inputs_dict["voxels"]
        voxel_features = self.pts_voxel_encoder(
            voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"]
        )
        batch_size = voxel_dict["coors"][-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict["coors"], batch_size)
        return x

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x
