#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device cuda \
  --processes 16 \
  --batch_size 16 \
  --root /home/trong/Downloads/Dataset/VAD/iitb \
  --save_root out \
  --del_prev_result false \
  --fn_name stage_two


# Run in vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out
