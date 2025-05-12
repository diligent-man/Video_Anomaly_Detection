#!/bin/sh
python3 ../../src/tools/preprocess.py\
    --device cpu \
    --processes 16 \
    --batch_size 16 \
    --root /home/trong/Downloads/Dataset/VAD/tmp_test/IITB/ \
    --cpu_ratio 0.5 \
    --save_root out \
    --wait_time 10 \
    --run_async true\
    --fn_name stage_one
