PATH_TO_YOUR_PIPELINE_CONFIG=/home/malte/Schreibtisch/models/research/object_detection/samples/configs/ssd_inception_v1_kitti.config
PATH_TO_TRAIN_DIR=./inception_v1_kitti/
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
