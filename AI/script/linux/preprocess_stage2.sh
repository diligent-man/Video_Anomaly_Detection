#!/bin/sh
# Run in alienware
python3 ../../src/tools/preprocess.py\
    --device cpu \
    --processes 12 \
    --batch_size 12 \
    --root /home/trong/Downloads/Dataset/VAD/tmp_test/out/UCF/ \
    --cpu_ratio 0.0 \
    --save_root ./ \
    --del_prev_result true \
    --fn_name stage_two
