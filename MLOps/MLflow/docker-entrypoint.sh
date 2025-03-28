#!/bin/bash
mlflow server --backend-store-uri "$BACKEND_STORE_URI" --registry-store-uri "$REGISTRY_STORE_URI" --app-name basic-auth --port "$MLFLOW_PORT" --host 0.0.0.0 --gunicorn-opts="--timeout=3600"
python3 -m mlflow.server.auth db upgrade --url postgresql://root:Root123!@postgres:5432/auth
