all_patients=('RARP01' 'RARP02' 'RARP03' 'RARP04' 'RARP06' 'RARP07' 'RARP08' 'RARP09' 'RARP10' 'RARP11' 'RARP12' 'RARP13' 'RARP15' 'RARP16' 'RARP17' 'RARP18' 'RARP19' 'RARP20' 'RARP21' 'RARP22' 'RARP23' 'RARP25' 'RARP26' 'RARP27' 'RARP28' 'RARP29' 'RARP30' 'RARP31' 'RARP32' 'RARP33' 'RARP34' 'RARP35' 'RARP36' 'RARP37' 'RARP38' 'RARP40' 'RARP41' 'RARP43' 'RARP44' 'RARP45' 'RARP46' 'RARP47' 'RARP48' 'RARP49' 'RARP50' 'RARP59' 'RARP61' 'RARP62' 'RARP64' 'RARP65')
n_train=35
n_val=8
n_test=7
TRAIN_FOLDS=()
VAL_FOLDS=()
TEST_FOLDS=()
GT_TRAIN_FOLDS=()
GT_VAL_FOLDS=()
GT_TEST_FOLDS=()
for pat in ${all_patients[@]:0:$n_train}; do
    echo $pat
    TRAIN_FOLDS+=(${pat}.csv)
    GT_TRAIN_FOLDS+=(${pat}_coco.json)
done
for pat in ${all_patients[@]:$n_train:$n_val}; do
    echo $pat
    VAL_FOLDS+=(${pat}.csv)
    GT_VAL_FOLDS+=(${pat}_coco.json)
done
for pat in ${all_patients[@]:$(($n_train + $n_val)):$n_test}; do
    echo $pat
    TEST_FOLDS+=(${pat}.csv)
    GT_TEST_FOLDS+=(${pat}_coco.json)
done
echo "train: ${TRAIN_FOLDS[@]}"
echo "val: ${VAL_FOLDS[@]}"
echo "test: ${TEST_FOLDS[@]}"

