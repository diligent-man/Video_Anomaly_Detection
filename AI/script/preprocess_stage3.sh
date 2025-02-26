#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --root /home/trong/Downloads/Dataset/VAD/iitb \
  --save_root out \
  --del_prev_result true \
  --vid_ext avi \
  --fn_name stage_three


# Run in vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out
