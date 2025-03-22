#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device both \
  --processes 8 \
  --batch_size 8 \
  --root /home/trong/Downloads/Dataset/VAD/UCF/ \
  --cpu_ratio 0.5 \
  --save_root out \
  --wait_time 30 \
  --run_async false\
  --fn_name stage_one

# Error notes
# UBI-FIGHT dataset
# Error:
#   [h264 @ 0x5cfefea3ca80] left block unavailable for requested intra4x4 mode -1
#   [h264 @ 0x5cfefea3ca80] error while decoding MB 0 19, bytestream 14868
#
#
# UCF dataset
# UCF error file:
#    unlabeled/anomaly/arrest/Arrest050_x264.mp4
#    unlabeled/anomaly/assault/Assault017_x264.mp4
#    unlabeled/anomaly/robbery/Robbery077_x264.mp4
# Error:
#   Expected string index (e.g. 'a'); got None
