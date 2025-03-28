# Run in alienware
# How to use: Copy and paste to py-env-enabled terminal
py ../../src/tools/preprocess.py `
    --device both `
    --processes 32 `
    --batch_size 64 `
    --root "D:\Dataset\VAD\crawled_data" `
    --cpu_ratio 0.5 `
    --save_root out `
    --wait_time 30 `
    --run_async true `
    --fn_name stage_one

# Error notes
# Crawled dataset
# Error:
#   Assertion pkt failed at src/fftools/ffmpeg_dec.c:597