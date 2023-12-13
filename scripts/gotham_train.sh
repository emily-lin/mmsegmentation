#!/bin/bash
if [ "$1" = "" ]
then
  echo "Usage: $0 <checkpoint_saving_directory_name>"
  exit
fi

python tools/train.py configs/swin/20231130_gotham_track_swin_base.py --resume --work-dir="./work_dirs/$1"

# Command to save visualization.
#VIS_DIR=/tmp/vis_dir
#python tools/test.py configs/swin/swin-tiny-track-512x512.py ./work_dirs/20220627_i160k/best_mDice_iter_128000.pth --show-dir=$VIS_DIR
