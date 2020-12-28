#!/usr/bin/env bash
export PYTHONPATH=/mnt/nvme/siamf/:$PYTHONPATH
cd ../experiments/$1
x_pad=0.5
z_pad=1.2

if [ ! $3 ];then
    GPUID='0'
else
    GPUID=$3
fi


if [ $2 == 'eval' ];then
    python ../../tools/eval.py 	 \
        --tracker_path ./results \
        --dataset OTB100        \
        --num 1 		 \
        --tracker_prefix 'model'
else
    for lambdav in 0.01 0.1 1.0 10.0 100.0
    do
        for lrv in `seq 0.1 0.05 0.3`
        do
            prefix="_xpad_$x_pad""_zpad_$z_pad""_lambdav_$lambdav""_lrv_$lrv"
            echo $prefix
            python -u ../../tools/test.py 	\
                --snapshot model.pth 	\
                --prefix $prefix \
                --lambda_v $lambdav \
                --lr_v $lrv \
                --dataset OTB100 	\
                --config config.yaml \
                --gpuid $GPUID
        done
    done
fi
cd ../../