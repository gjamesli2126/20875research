#! /bin/bash -x

APP=vp
TYPE=Hybrid_Sched
INPUT_DIR=../../../input/pc/
#INPUT_FILES=(${INPUT_DIR}covtype.7d ${INPUT_DIR}mnist.7d ${INPUT_DIR}random.7d ${INPUT_DIR}geocity.txt)
#INPUT_DIM=(7 7 7 3)
INPUT_FILES=(${INPUT_DIR}geocity.txt)
INPUT_DIM=(2)
WARP_DIR=../../../results/warp_length/${APP}/
TIME_DIR=../../../results/runtime_35/${APP}/
TMP_DIR=../../../results/tmp/
SCRIPT_DIR=../../../scripts/
NUM_RUNS=15
npoints=200000
splice_depth=4

MAKE_ARGS=()
for arg in $*
do
    MAKE_ARGS[${#MAKE_ARGS}]=$arg
    if [[ $arg = "TRACK_TRAVERSALS=1" ]]
    then
        NUM_RUNS=1
    fi
done

((j=0))
for file in ${INPUT_FILES[*]}
do
    make clean;
    make ${MAKE_ARGS[*]} DIM=${INPUT_DIM[$j]} SPLICE_DEPTH=${splice_depth}
    f=$(basename $file)

    echo "deleting tmp files in app=$APP type=$TYPE input=$f..."
    rm -rf ${TMP_DIR}${APP}_${TYPE}_${f}_unsorted.stats
    rm -rf ${TMP_DIR}${APP}_${TYPE}_${f}_unsorted.out
    rm -rf ${TIME_DIR}${APP}_${TYPE}_${f}_unsorted.time

    for ((i = 0; i < NUM_RUNS; i ++))
    do
        echo "$i th iteration for time test in app=$APP type=$TYPE input=$f..."
        ./vptree $file $npoints 2>&1 | ${SCRIPT_DIR}collect.py ${TMP_DIR}${APP}_${TYPE}_${f}_unsorted.stats 2>&1 >> ${TMP_DIR}${APP}_${TYPE}_${f}_unsorted.out
    done

    echo "generating result for time test in app=$APP type=$TYPE input=$f..."
    python ${SCRIPT_DIR}show_results.py ${TMP_DIR}${APP}_${TYPE}_${f}_unsorted.stats 2>&1 >> ${TIME_DIR}${APP}_${TYPE}_${f}_unsorted.time

    ((j++))
done

