#!/usr/bin/env bash
export PYTHONPATH=/mnt/nvme/siamf/:$PYTHONPATH
cd ../experiments/$1

if [ $1 == 'dsiamrpn_alex_dwxcorr' ];then
    x_pad=0.5
    z_pad=1.2
    lambdav=1.0
    lrv=0.16
elif [ $1 == 'dsiamrpn_mobilev2_l234_dwxcorr' ];then
    x_pad=0.5
    z_pad=1.2
    lambdav=20.0
    lrv=0.06
elif [ $1 == 'dsiamrpn_r50_l234_dwxcorr_otb' ];then
    x_pad=0.5
    z_pad=1.2
    lambdav=10.0
    lrv=0.01
fi

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
    for interval in 0.01 0.1 1.0 10.0 100.0
    do
        prefix="_xpad_$x_pad""_zpad_$z_pad""_lambdav_$lambdav""_lrv_$lrv""_interval_$interval"
        echo $prefix
        python -u ../../tools/test.py 	\
            --snapshot model.pth 	\
            --prefix $prefix \
            --lambda_v $lambdav \
            --lr_v $lrv \
            --dataset OTB100 	\
            --config config.yaml \
            --sfinterval $interval \
            --gpuid $GPUID
    done
fi
cd ../../