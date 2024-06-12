from mmdet3d.testing import create_detector_inputs, get_detector_cfg, setup_seed


def get_example_data(model, num_gt_instance=2, points_feat_dim=7):
    packed_inputs = create_detector_inputs(
        num_gt_instance=num_gt_instance, points_feat_dim=points_feat_dim, with_img=True
    )
    data = model.data_preprocessor(packed_inputs, True)
    return data
