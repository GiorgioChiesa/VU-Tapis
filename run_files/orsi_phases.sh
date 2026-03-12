NAME="Orsi-finetune_all"
# Experiment setup
TRAIN_FOLDS="['RARP01.csv','RARP07.csv','RARP18.csv','RARP23.csv','RARP29.csv','RARP34.csv','RARP40.csv','RARP46.csv','RARP59.csv','RARP02.csv','RARP08.csv','RARP13.csv','RARP19.csv','RARP25.csv','RARP30.csv','RARP35.csv','RARP41.csv','RARP47.csv','RARP61.csv','RARP03.csv','RARP09.csv','RARP15.csv','RARP20.csv','RARP26.csv','RARP31.csv','RARP43.csv','RARP48.csv','RARP62.csv','RARP04.csv','RARP10.csv','RARP16.csv','RARP21.csv','RARP27.csv','RARP32.csv','RARP37.csv','RARP44.csv','RARP49.csv','RARP64.csv']"
TEST_FOLDS="['RARP06.csv','RARP11.csv','RARP17.csv','RARP22.csv','RARP28.csv','RARP33.csv','RARP38.csv','RARP45.csv','RARP65.csv']"
GT_TRAIN_FOLDS="['RARP01_coco.json','RARP07_coco.json','RARP18_coco.json','RARP23_coco.json','RARP29_coco.json','RARP34_coco.json','RARP40_coco.json','RARP46_coco.json','RARP59_coco.json','RARP02_coco.json','RARP08_coco.json','RARP13_coco.json','RARP19_coco.json','RARP25_coco.json','RARP30_coco.json','RARP35_coco.json','RARP41_coco.json','RARP47_coco.json','RARP61_coco.json','RARP03_coco.json','RARP09_coco.json','RARP15_coco.json','RARP20_coco.json','RARP26_coco.json','RARP31_coco.json','RARP43_coco.json','RARP48_coco.json','RARP62_coco.json','RARP04_coco.json','RARP10_coco.json','RARP16_coco.json','RARP21_coco.json','RARP27_coco.json','RARP32_coco.json','RARP37_coco.json','RARP44_coco.json','RARP49_coco.json','RARP64_coco.json']"
GT_TEST_FOLDS="['RARP06_coco.json','RARP11_coco.json','RARP17_coco.json','RARP22_coco.json','RARP28_coco.json','RARP33_coco.json','RARP38_coco.json','RARP45_coco.json','RARP65_coco.json']"
EXP_PREFIX=$NAME  #costumize
TASK="PHASES"
ARCH="TAPIS"
GPUIDS="2,3"
#-------------------------
DATASET="orsi"
CONFIG_PATH="configs/Orsi/$ARCH/TAPIS_PHASES.yaml"
OUTPUT_DIR="/home/gchie/workspace/VU-Tapis/outputs/"$DATASET"/"$TASK"/"$NAME"/totale"

#Change this variables if data is not located in ./data
FRAME_DIR="/home/gchie/workspace/nas_private/data/orsi"
FRAME_LIST="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/frame_lists"
ANNOT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations"
COCO_ANN_PATH="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/annotations/grasp_long-term_"$TEST_FOLD".json"
CHECKPOINT="/scratch/Video_Understanding/GraSP/TAPIS/data/"$DATASET"/pretrained_models/fold1/"$TASK".pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/tapis:$PYTHONPATH
export PYTHONPATH=/home/chiesa/scratch/Video_Understanding/GraSP/TAPIS/region_proposals:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$GPUIDS

# export $(cut -f1 .secret/.export_vars.txt)
# echo "Using WANDB_API_KEY: $WANDB_API"
# wandb login --relogin --key $WANDB_API

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
WANDB_ENABLE True \
NAME $NAME \
GPUIDS "[$GPUIDS]" \
TRAIN.FREEZE_ENCODER False \
OUTPUT_DIR $OUTPUT_DIR \
ENDOVIS_DATASET.FRAME_DIR /home/gchie/workspace/nas_private/data/orsi \
ENDOVIS_DATASET.FRAME_LIST_DIR /home/gchie/workspace/nas_private/data/coco \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLDS \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLDS \
ENDOVIS_DATASET.ANNOTATION_DIR /home/gchie/workspace/nas_private/data/coco \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON $GT_TRAIN_FOLDS \
ENDOVIS_DATASET.TEST_GT_BOX_JSON $GT_TEST_FOLDS \
ENDOVIS_DATASET.TEST_COCO_ANNS /home/gchie/workspace/nas_private/data/coco/all_merged.json 