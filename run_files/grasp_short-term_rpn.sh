# Experiment setup
TRAIN_FOLD="train" # or fold1, fold2
TEST_FOLD="test" # or fold1, fold2
EXP_PREFIX="Prova" # costumize
TASK="SHORT"
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
RPN_CHECKPOINT="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/pretrained_models/"$TRAIN_FOLD"/SEGMENTATION_BASELINE/swinl.pth"
RPN_CONFIG="/scratch/Video_Understanding/GraSP/TAPIS/region_proposals/configs/grasp/GraSP_SwinL_regions.yaml"
#-------------------------
# Run experiment

export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/tapis:$PYTHONPATH
export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/region_proposals:$PYTHONPATH

mkdir -p $OUTPUT_DIR


# removed # TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \ from original
python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TEST.ENABLE True \
TRAIN.ENABLE True \
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
TRAIN.BATCH_SIZE 56 \
TEST.BATCH_SIZE 56 \
DATA.TRAIN_CROP_SIZE 224 \
DATA.TRAIN_CROP_SIZE_LARGE 356 \
DATA.TEST_CROP_SIZE 224 \
DATA.TEST_CROP_SIZE_LARGE 356 \
OUTPUT_DIR $OUTPUT_DIR \
FEATURES.USE_RPN True \
FEATURES.RPN_CFG_PATH $RPN_CONFIG \
FEATURES.RPN_CHECKPOINT $RPN_CHECKPOINT \
FEATURES.PRECALCULATE_TEST True # Switch to false to avoid using precalculated regions for inference