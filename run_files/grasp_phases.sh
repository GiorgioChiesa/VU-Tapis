sh install.sh
export $(grep -v '^#' .secret/.export_vars.txt | xargs)
# Experiment setup
TRAIN_FOLD="train" # or fold1, train
TEST_FOLD="test" # or fold2, test
EXP_PREFIX=$NAME # costumize
TASK="PHASES"
ARCH="TAPIS"

#-------------------------
DATASET="GraSP"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="/scratch/Video_Understanding/GraSP/TAPIS/configs/"$DATASET"/"$ARCH"/"$ARCH"_"$TASK".yaml"
OUTPUT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME

# Change this variables if data is not located in ./data
FRAME_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/frames"
FRAME_LIST="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/frame_lists"
ANNOT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations"
COCO_ANN_PATH="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations/grasp_long-term_"$TEST_FOLD".json"
CHECKPOINT="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/pretrained_models/fold1/"$TASK".pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/tapis:$PYTHONPATH
export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/region_proposals:$PYTHONPATH

# export $(cut -f1 .secret/.export_vars.txt)
# echo "Using WANDB_API_KEY: $WANDB_API"
# wandb login --relogin --key $WANDB_API

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TEST.ENABLE True \
TRAIN.ENABLE True \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "grasp_long-term_"$TRAIN_FOLD".json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON "grasp_long-term_"$TEST_FOLD".json" \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
TRAIN.BATCH_SIZE 96 \
TEST.BATCH_SIZE 96 \
OUTPUT_DIR $OUTPUT_DIR 