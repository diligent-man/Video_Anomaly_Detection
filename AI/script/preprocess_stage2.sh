#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device cuda \
  --processes 8 \
  --batch_size 8 \
  --root /home/trong/Downloads/Dataset/VAD/iitb \
  --cpu_ratio 0.25 \
  --save_root out \
  --del_prev_result true \
  --fn_name stage_two


# Run in vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out
