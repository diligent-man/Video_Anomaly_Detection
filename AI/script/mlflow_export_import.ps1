#!/bin/sh
py ../src/tools/mlflow_export_import.py \
    --src_run_id a73c5befce0f42ada142ab1cd64e5bb0 \
    --dst_exp_name AIP391 \
    --src_uri http://loclahost:5000 \
    --dst_uri http://113.23.3.195:5000 \
    --username root \
    --password Root123!
