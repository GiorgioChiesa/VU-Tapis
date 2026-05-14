# Experiment setup
all_patients=('RARP01' 'RARP02' 'RARP03' 'RARP04' 'RARP06' 'RARP07' 'RARP08' 'RARP09' 'RARP10' 'RARP11' 'RARP12' 'RARP13' 'RARP15' 'RARP16' 'RARP17' 'RARP18' 'RARP19' 'RARP20' 'RARP21' 'RARP22' 'RARP23' 'RARP25' 'RARP26' 'RARP27' 'RARP28' 'RARP29' 'RARP30' 'RARP31' 'RARP32' 'RARP33' 'RARP34' 'RARP35' 'RARP36' 'RARP37' 'RARP38' 'RARP40' 'RARP41' 'RARP43' 'RARP44' 'RARP45' 'RARP46' 'RARP47' 'RARP48' 'RARP49' 'RARP50' 'RARP59' 'RARP61' 'RARP62' 'RARP64' 'RARP65')
n_train=43
n_val=6
n_test=1
TRAIN_FOLDS_STR=""
VAL_FOLDS_STR=""
TEST_FOLDS_STR=""
for pat in ${all_patients[@]:0:$n_train}; do
    if [ -n "$TRAIN_FOLDS_STR" ]; then
        TRAIN_FOLDS_STR="${TRAIN_FOLDS_STR},'${pat}.csv'"
    else
        TRAIN_FOLDS_STR="'${pat}.csv'"
    fi
done
for pat in ${all_patients[@]:$n_train:$n_val}; do
    if [ -n "$VAL_FOLDS_STR" ]; then
        VAL_FOLDS_STR="${VAL_FOLDS_STR},'${pat}.csv'"
    else
        VAL_FOLDS_STR="'${pat}.csv'"
    fi
done
for pat in ${all_patients[@]:$((n_train + n_val)):$n_test}; do
    if [ -n "$TEST_FOLDS_STR" ]; then
        TEST_FOLDS_STR="${TEST_FOLDS_STR},'${pat}.csv'"
    else
        TEST_FOLDS_STR="'${pat}.csv'"
    fi
done
echo "TRAIN: [$TRAIN_FOLDS_STR]"
echo "VAL: [$VAL_FOLDS_STR]"
echo "TEST: [$TEST_FOLDS_STR]"

NAME="actions_run1"
GPUIDS="0"

DATASET="orsi"
TASK="ACTIONS"
ARCH="TAPIS"
CONFIG_PATH="configs/Orsi/$ARCH/TAPIS_ACTIONS_ORSI.yaml"
OUTPUT_DIR="outputs/"$DATASET"/"$TASK"/"$NAME"/totale"

FRAME_DIR="/data/orsi_tensors"
FRAME_LIST="/data/coco"
ANNOT_DIR="/data/coco"
COCO_ANN_PATH="/data/coco/all_merged.json"

export PYTHONPATH=/workspaces/thesis_giorgiochiesa/src/VU-Tapis/tapis:$PYTHONPATH
export PYTHONPATH=/workspaces/thesis_giorgiochiesa/src/VU-Tapis/detectron2:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$GPUIDS

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$GPUIDS python -B tools/run_net.py \
--cfg $CONFIG_PATH \
WANDB_ENABLE False \
NAME $NAME \
GPUIDS "[$GPUIDS]" \
TRAIN.ACCUM_STEPS 10 \
TRAIN.BATCH_SIZE 12 \
VAL.BATCH_SIZE 24 \
SOLVER.MAX_ITER 7000 \
TRAIN.FREEZE_ENCODER False \
OUTPUT_DIR $OUTPUT_DIR \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS "[$TRAIN_FOLDS_STR]" \
ENDOVIS_DATASET.VAL_LISTS "[$VAL_FOLDS_STR]" \
ENDOVIS_DATASET.TEST_LISTS "[$TEST_FOLDS_STR]" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.VAL_COCO_ANNS $COCO_ANN_PATH \
TEST.ENABLE False
