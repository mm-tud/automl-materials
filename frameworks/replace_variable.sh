#!/bin/bash

# Watch out, edits multiple files without backup 
find . | grep run.py | xargs sed -i "s/OUTER_SPLITS = .*/OUTER_SPLITS = 1/g"
