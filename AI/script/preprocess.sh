#!/bin/sh
python3 ../src/tools/preprocess_video.py \
  --device both \
  --processes 16 \
  --batch_size 16 \
  --root ../dataset/ubi-fight \
  --save_root ../dataset/out


# Run in Vostro
# python3 ../src/tools/preprocess_video.py \
#   --device both --processes 16 \
#   --batch_size 16 \
#   --root /media/trong/Backup/Dataset/VAD/iitb \
#   --save_root /media/trong/Backup/Dataset/VAD/out

