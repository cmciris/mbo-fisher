#!/bin/bash

exp=${1:-"rembo"}
env=${2:-"hopper"}
runs=${3:-1}
device=${4:-0}

echo "Run ${exp}-${env} ${runs} times on cuda:${device}"

for i in `seq ${runs}`
do
CUDA_VISIBLE_DEVICES=${device} python main.py -c ${exp}-${env}
done