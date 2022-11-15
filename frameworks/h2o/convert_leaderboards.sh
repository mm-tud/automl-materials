#!/bin/bash

# feed all "leaderboard.txt" found in workdir into array
mapfile array < <(find workdir -name "leaderboard.txt" )

# for every file
for i in "${array[@]}"
do
    filename="$i"
    target="$(dirname $i)/leaderboard.csv"
    
    # create copy
    cp $filename $target 

    # remove last three (blank) lines
    head -q -n-3 $target > dummy
    mv dummy $target

    # remove second line (separator)
    sed '2d' $target > dummy
    mv dummy $target

    # replace all whitespaces by comma
    sed 's/ \+/,/g' $target > dummy
    mv dummy $target
done
