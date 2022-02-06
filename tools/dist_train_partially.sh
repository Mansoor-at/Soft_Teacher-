#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
export NGPUS=2
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$PORT\
	 $(dirname "$0")/train.py configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py --launcher pytorch \
        --gpus 2 --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
else
    CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py --launcher pytorch \
        --gpus 2 --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
fi
