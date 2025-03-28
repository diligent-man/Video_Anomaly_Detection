# How to use: Copy and paste to py-env-enabled terminal
# Run in alienware
py ../../src/tools/preprocess.py `
    --device cpu `
    --processes 12 `
    --batch_size 12 `
    --root "D:\Dataset\VAD\crawled_data" `
    --cpu_ratio 0.5 `
    --save_root out `
    --del_prev_result true `
    --fn_name stage_two
