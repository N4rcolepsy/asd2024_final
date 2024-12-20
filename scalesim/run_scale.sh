# Config File Path
TOPOLOGY_FILE="../topologies/mlperf/Resnet50.csv"
CONFIG_FILE="../configs/csc/eyeriss_os_32_64.cfg"
LOG_DIR="../test_runs/csc"
INPUT_TYPE="conv"

#Run ScaleSim
python scale.py \
    -t $TOPOLOGY_FILE \
    -c $CONFIG_FILE \
    -p $LOG_DIR \
    -i $INPUT_TYPE \