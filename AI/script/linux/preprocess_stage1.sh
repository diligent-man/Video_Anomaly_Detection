#!/bin/sh
# Run in alienware
python3 ../../src/tools/preprocess.py\
    --device cpu \
    --processes 16 \
    --batch_size 16 \
    --root /home/trong/Downloads/Dataset/VAD/crawled_data/ \
    --cpu_ratio 0.5 \
    --save_root out \
    --wait_time 30 \
    --run_async true\
    --fn_name stage_one

# Error notes
# UBI-FIGHT dataset
# Error:
#   [h264 @ 0x5cfefea3ca80] left block unavailable for requested intra4x4 mode -1
#   [h264 @ 0x5cfefea3ca80] error while decoding MB 0 19, bytestream 14868
#
#
# Crawled dataset
# Error:
#   Assertion pkt failed at src/fftools/ffmpeg_dec.c:597
