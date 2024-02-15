#!/bin/bash
if [ "$1" = "" ]
then
  echo "Usage: $0 <path to load the checkpoint.>"
  exit
fi

python tools/test.py configs/swin/20231212_batch60_adjustlr.py $1

# Command to save visualization.
#VIS_DIR=/tmp/vis_dir
#python tools/test.py configs/swin/swin-tiny-track-512x512.py ./work_dirs/20220627_i160k/best_mDice_iter_128000.pth --show-dir=$VIS_DIR
