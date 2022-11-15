#!/bin/bash

ROOT_DIR=$(pwd | rev | cut -d'/' -f2- | rev)

find $ROOT_DIR/frameworks/*/workdir/* -maxdepth 0 -type d -exec rm -rf {} +
