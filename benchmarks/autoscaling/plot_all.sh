#!/bin/bash

parent_dir=$1
if [ -z "$parent_dir" ]; then
    echo "Usage: $0 <parent_dir>"
    exit 1
fi

all_sub_dirs=$(ls -d $parent_dir/*/)
for subdir in $all_sub_dirs; do
    echo $subdir
    python plot/plot-output.py $subdir
done