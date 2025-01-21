#!/bin/bash
#mlflow server --backend-store-uri "$BACKEND_STORE_URI" --registry-store-uri "$REGISTRY_STORE_URI" --default-artifact-root "$DEFAULT_ARTIFACT_ROOT" --artifacts-destination "$ARTIFACTS_DESTINATION" --app-name basic-auth --port "$MLFLOW_PORT" --host 0.0.0.0
mlflow server --backend-store-uri "$BACKEND_STORE_URI" --registry-store-uri "$REGISTRY_STORE_URI" --app-name basic-auth --port "$MLFLOW_PORT" --host 0.0.0.0

python3 -m mlflow.server.auth db upgrade --url postgresql://root:Root123!@postgres:5432/auth
