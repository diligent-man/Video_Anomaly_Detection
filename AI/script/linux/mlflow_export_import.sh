#!/bin/sh
python3 ../../src/tools/mlflow_export_import.py \
    --experiment_name AIP391 \
    --run_id <run_id> \
    --backend_store_uri <path_to_mlflow_dor> \
    --src_uri http://localhost:5000 \
    --dst_uri http://<ip>:5000 \
    --username root \
    --password Root123! && \
pkill -f gunicorn
