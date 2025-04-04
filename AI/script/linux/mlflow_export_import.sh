#!/bin/sh
python3 ../../src/tools/mlflow_export_import.py \
    --experiment_name AIP391 \
    --run_id 4d35181acbdd46839b4ecf69060f5856 \
    --backend_store_uri /home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/AIP391/single/train/run/Mlflow \
    --src_uri http://localhost:5000 \
    --dst_uri http://42.119.174.56:5000 \
    --username root \
    --password Root123!
