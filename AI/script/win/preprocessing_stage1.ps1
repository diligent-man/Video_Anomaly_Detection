# Run in alienware
# How to use: Copy and paste to py-env-enabled terminal
py ../../src/tools/preprocess.py `
    --device both `
    --processes 32 `
    --batch_size 32 `
    --root "D:\Dataset\VAD\filtered\UCF" `
    --cpu_ratio 0.7 `
    --save_root "../preprocessed" `
    --wait_time 30 `
    --run_async false `
    --fn_name stage_one

# Error notes
# Crawled dataset
# Error:
#   Assertion pkt failed at src/fftools/ffmpeg_dec.c:597 (crawled_assault_000013.mp4)
#   -> Solution: process with Handbrake app