#!/bin/bash

ARRAY=$1
SIZE=$2

# Config File Path
TOPOLOGY_FILE="../topologies/mlperf/Resnet50.csv"
CONFIG_FILE="../configs/baseline/eyeriss_os_${ARRAY}_${SIZE}.cfg"
LOG_DIR="../test_runs/baseline"
INPUT_TYPE="conv"

#Run ScaleSim
python ./scale.py \
    -t $TOPOLOGY_FILE \
    -c $CONFIG_FILE \
    -p $LOG_DIR \
    -i $INPUT_TYPE \