from mmengine import DefaultScope
from mmdet3d.testing import create_detector_inputs, get_detector_cfg, setup_seed


def get_example_data(model, num_gt_instance=2, points_feat_dim=7):
    DefaultScope.get_instance("prune", scope_name="mmdet3d")
    packed_inputs = create_detector_inputs(
        num_gt_instance=num_gt_instance, points_feat_dim=points_feat_dim
    )
    data = model.data_preprocessor(packed_inputs, True)
    return data
