#!/bin/sh
python3 ../../src/tools/mlflow_export_import.py \
    --experiment_name AIP391 \
    --run_id ade2ba9302374d17821620e17114ebb1 \
    --backend_store_uri /home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/AIP391/single/train/input_2d/Mlflow \
    --src_uri http://localhost:5000 \
    --dst_uri http://192.168.100.208:5000 \
    --username root \
    --password Root123!
