#!/bin/bash

docker build --pull . -t $(basename $(pwd))_fcmm:latest
