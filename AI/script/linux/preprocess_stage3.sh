#!/bin/sh
# Run in alienware
python3 ../../src/tools/preprocess.py\
  --root /home/trong/Downloads/Dataset/VAD/IITB/ \
  --save_root out \
  --del_prev_result true \
  --vid_ext avi \
  --fn_name stage_three
