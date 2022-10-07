#!/bin/bash

ROOT_DIR=$(pwd | rev | cut -d'/' -f2- | rev)

cd $ROOT_DIR/frameworks/autosklearn
./start.sh
cd $ROOT_DIR/frameworks/h2o
./start.sh
cd $ROOT_DIR/frameworks/mljar
./start.sh
cd $ROOT_DIR/frameworks/tpot
./start.sh
