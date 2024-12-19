# Config File Path
TOPOLOGY_FILE="../topologies/mlperf/Resnet50.csv"
CONFIG_FILE="../configs/eyeriss_os_pe32.cfg"
LOG_DIR="../test_runs"
INPUT_TYPE="conv"

#Run ScaleSim
python scale.py \
    -t $TOPOLOGY_FILE \
    -c $CONFIG_FILE \
    -p $LOG_DIR \
    -i $INPUT_TYPE \