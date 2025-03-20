#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device both \
  --processes 32 \
  --batch_size 32 \
  --root /home/trong/Downloads/Dataset/VAD/ucf/ \
  --cpu_ratio 0.5 \
  --save_root out \
  --run_async true\
  --fn_name stage_one


# Run in Vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out

# Note
#UCF error file:
#    unlabeled/anomaly/arrest/Arrest050_x264.mp4
#    unlabeled/anomaly/assault/Assault017_x264.mp4
#    unlabeled/anomaly/robbery/Robbery077_x264.mp4
#Cause: Expected string index (e.g. 'a'); got None
