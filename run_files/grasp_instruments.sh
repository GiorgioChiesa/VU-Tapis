# Experiment setup
TRAIN_FOLD="train" # or fold1, fold2
TEST_FOLD="test" # or fold1, fold2
EXP_PREFIX="test_run" # costumize
TASK="INSTRUMENTS"
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
FF_TRAIN="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/features/"$TRAIN_FOLD"_train_region_features.pth" 
FF_TEST="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/features/"$TEST_FOLD"_val_region_features.pth"
CHECKPOINT="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/pretrained_models/"$TRAIN_FOLD"/"$TASK".pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/tapis:$PYTHONPATH
export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/region_proposals:$PYTHONPATH

export $(cut -f1 .secret/.export_vars.txt)
echo "Using WANDB_API_KEY: $WANDB_API"
wandb login --relogin --key $WANDB_API

# # Uncomment to calculate region proposals on the fly
# export PYTHONPATH=/home/nayobi/Endovis/GraSP/TAPIS/region_proposals:$PYTHONPATH

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TEST.ENABLE True \
TRAIN.ENABLE False \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "grasp_short-term_"$TRAIN_FOLD".json" \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON $TRAIN_FOLD"_train_preds.json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON "grasp_short-term_"$TEST_FOLD".json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON $TEST_FOLD"_val_preds.json" \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.TEST_FEATURES_PATH $FF_TEST \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
OUTPUT_DIR $OUTPUT_DIR \
FEATURES.USE_RPN False # Switch to True to calculate region proposals on the fly (this makes training slower)