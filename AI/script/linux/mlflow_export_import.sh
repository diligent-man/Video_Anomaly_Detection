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
    --run_id aaff4465b324425b91421ee50a683603 \
    --backend_store_uri /home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/Mlflow \
    --src_uri http://localhost:5000 \
    --dst_uri http://42.113.51.159:5000 \
    --username root \
    --password Root123! && \
pkill -f gunicorn
