#!/bin/bash
# Config authentication in mlflow

cat << EOF > /usr/local/lib/python3.10/site-packages/mlflow/server/auth/basic_auth.ini
[mlflow]
default_permission = READ
database_uri = postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${POSTGRES_PORT}/auth
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

# Spin up mlflow
mlflow server \
--backend-store-uri ${BACKEND_STORE_URI} \
--registry-store-uri ${REGISTRY_STORE_URI} \
--default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
--artifacts-destination ${ARTIFACTS_DESTINATION} \
--app-name basic-auth \
--port ${MLFLOW_PORT} \
--host 0.0.0.0

python3 -m mlflow.server.auth db upgrade --url postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${POSTGRES_PORT}
