#!/usr/bin/env bash

CONFIG=$1
export NGPUS=2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --gpus 2 ${@:3}
