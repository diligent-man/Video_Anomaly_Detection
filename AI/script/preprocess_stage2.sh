#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device cpu \
  --processes 8 \
  --batch_size 8 \
  --root /home/trong/Downloads/Dataset/VAD/ubi-fight/ \
  --cpu_ratio 0.5 \
  --save_root out \
  --wait-time 30 \
  --del_prev_result true \
  --fn_name stage_two


# Run in vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out
