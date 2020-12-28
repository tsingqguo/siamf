#!/usr/bin/env bash
export PYTHONPATH=/mnt/nvme/siamf/:$PYTHONPATH
cd ../experiments/$1

lambdav=0.1
lrv=0.2

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
    for x_pad in `seq 0.1 0.1 0.5`
    do
        for z_pad in `seq 1.0 0.2 3.0`
        do
            prefix="_xpad_$x_pad""_zpad_$z_pad""_lambdav_$lambdav""_lrv_$lrv"
            echo $prefix
                python -u ../../tools/test.py 	\
                    --snapshot model.pth 	\
                    --prefix $prefix \
                    --xpad $x_pad \
                    --zpad $z_pad \
                    --lambda_v $lambdav \
                    --lr_v $lrv \
                    --dataset OTB100 	\
                    --config config.yaml \
                    --gpuid $GPUID
        done
    done
fi
cd ../../