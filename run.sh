CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/vod/bevfusion/camera.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/vod/bevfusion/lidar.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/vod/bevfusion/fusion.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/vod/bevfusion/radar.py 4
