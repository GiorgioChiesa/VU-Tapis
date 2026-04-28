sh install.sh
export $(grep -v '^#' .secret/.export_vars.txt | xargs)

# Experiment setup
TRAIN_FOLD="train" # or fold1, fold2
VAL_FOLD="val" # or fold1, fold2
EXP_PREFIX="val_run" # custom
TASK="LONG"
ARCH="TAPIS"
NAME="Phases_steps"
#-------------------------
DATASET="GraSP"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="/scratch/Video_Understanding/GraSP/TAPIS/configs/"$DATASET"/"$ARCH"/"$ARCH"_"$TASK".yaml"
OUTPUT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME

# Change this variables if data is not located in ./data
FRAME_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/frames"
FRAME_LIST="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/frame_lists"
ANNOT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations"
COCO_ANN_PATH="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations/grasp_long-term_"$VAL_FOLD".json"
CHECKPOINT="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/pretrained_models/"$TRAIN_FOLD"/"$TASK".pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/tapis:$PYTHONPATH
export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/region_proposals:$PYTHONPATH

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
VAL.ENABLE True \
TRAIN.ENABLE True \
WANDB_ENABLE True \
NAME $NAME \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.VAL_LISTS $VAL_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "grasp_long-term_"$TRAIN_FOLD".json" \
ENDOVIS_DATASET.VAL_GT_BOX_JSON "grasp_long-term_"$VAL_FOLD".json" \
ENDOVIS_DATASET.VAL_COCO_ANNS $COCO_ANN_PATH \

OUTPUT_DIR $OUTPUT_DIR 