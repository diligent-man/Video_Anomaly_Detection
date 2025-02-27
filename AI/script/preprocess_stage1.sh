#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device both \
  --processes 18 \
  --batch_size 18 \
  --root /home/trong/Downloads/Dataset/VAD/ubi-fight \
  --cpu_ratio 0.5 \
  --save_root out \
  --run_async false \
  --fn_name stage_one


# Run in Vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out
