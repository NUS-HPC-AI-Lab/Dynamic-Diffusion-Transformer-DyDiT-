#!/usr/bin/env sh


GPUS=${GPUS:-8}
PORT=$((12000 + $RANDOM % 20000))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        --use_env \
        ./sample_ddp.py \
        --model DiT-XL/2 \
        --image-size 256  \
        --ckpt dydit_0.7.pth \
        --sample-dir "./samples0.7" \
        --num-fid-samples 50000 


set -x
python evaluator.py ../../imagenet_fid/VIRTUAL_imagenet256_labeled.npz \
./samples0.7.npz
set +x

