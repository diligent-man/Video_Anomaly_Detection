#!/bin/sh
#python3 ../../src/tools/mlflow_export_import.py \
#    --experiment_name AIP391 \
#    --run_id <run_id> \
#    --backend_store_uri <path_to_mlflow_dor> \
#    --src_uri http://localhost:5000 \
#    --dst_uri http://<ip>:5000 \
#    --username root \
#    --password Root123! && \
#pkill -f gunicorn

python3 ../../src/tools/mlflow_export_import.py \
    --experiment_name AIP391 \
    --run_id 5764473c20cf4e969a11f01258b7a223 \
    --backend_store_uri /home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/AIP391/distillation/train/input_3D_student/Mlflow \
    --src_uri http://localhost:5000 \
    --dst_uri http://42.116.137.220:5000 \
    --username root \
    --password Root123! && \
pkill -f gunicorn
