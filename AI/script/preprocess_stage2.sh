#!/bin/sh
# Run in alienware
python3 ../src/tools/preprocess.py\
  --device cpu \
  --processes 6 \
  --batch_size 6 \
  --root /home/trong/Downloads/Dataset/VAD/UCF/ \
  --cpu_ratio 0.4 \
  --save_root out \
  --del_prev_result true \
  --fn_name stage_two
