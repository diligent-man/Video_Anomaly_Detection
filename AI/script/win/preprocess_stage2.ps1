# How to use: Copy and paste to py-env-enabled terminal
# Run in alienware
py ../../src/tools/preprocess.py `
    --device cpu `
    --processes 32 `
    --batch_size 32 `
    --root "D:\Dataset\VAD\train_test_split\UBI-FIGHT" `
    --include_labeled true `
    --cpu_ratio 0.5 `
    --save_root "./" `
    --del_prev_result true `
    --fn_name stage_two
