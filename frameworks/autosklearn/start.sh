#!/bin/bash

ROOT_DIR=$(pwd | rev | cut -d'/' -f3- | rev)
CONTAINER=$(basename $(pwd))_fcmm:latest
DATA_DIR=$ROOT_DIR/data
WORK_DIR=$(pwd)/workdir

echo $CONTAINER
echo $DATA_DIR
echo $WORK_DIR

docker build . -t $CONTAINER
docker run --cpuset-cpus 0-7 -v $DATA_DIR:/data -v $WORK_DIR:/workdir -u $(id -u):$(id -g) $CONTAINER
