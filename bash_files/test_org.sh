#!/usr/bin/env bash
export PYTHONPATH=/mnt/nvme/siamf/:$PYTHONPATH
cd ../experiments/$1

if [ ! $3 ];then
    DATASET=OTB100
else
    DATASET=$3
fi

if [ ! $4 ];then
    GPUID='0'
else
    GPUID=$4
fi

if [ ! $5 ];then
    CFFEAT=-1
else
    CFFEAT=$5
fi

echo $DATASET

if [ $2 == 'eval' ];then
    python ../../tools/eval.py 	 \
        --tracker_path ./results \
        --dataset $DATASET        \
        --num 1 		 \
        --tracker_prefix 'model' \
        --gpuid $GPUID
else
    python -u ../../tools/test.py 	\
        --snapshot model.pth 	\
        --dataset $DATASET 	\
        --config config.yaml \
        --gpuid $GPUID \
        --cffeat $CFFEAT
fi

cd ../../