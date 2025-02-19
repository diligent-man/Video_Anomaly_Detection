#!/bin/bash
# Generate .lakefs.yaml inside image prior to spin up service.
# Must be run in conjunction with docker compose file
# Use bash file so as to make use of ENV var in .env

cat << EOF > $HOME/.lakefs.yaml
---
listen_address: "0.0.0.0:${LAKEFS_PORT}"

logging:
    format: text
    level: INFO
    output: "-"

database:
    type: "postgres"
    postgres:
        connection_string: "postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${POSTGRES_PORT}/lakefs"
        max_open_connections: 25
        max_idle_connections: 25
        connection_max_lifetime: 5m

blockstore:
    type: "s3"
    s3:
        region: ""
        discover_bucket_region: false
        disable_pre_signed: true  # not know reason but it enable client-server sync

        # Use path-style S3 url in lieu of http(s)://
        force_path_style: true

        # MinIO endpoint
        credentials:
            access_key_id: ${MINIO_ROOT_USER}
            secret_access_key: ${MINIO_ROOT_PASSWORD}

        # endpoint is specified in docker compose instead in this


auth:
    login_duration: "24h"
    login_max_duration: "24h"
    encrypt:
        secret_key: "10a718b3f285d89c36e9864494cdd1507f3bc85b342df24736ea81f9a1134bcc"

# Only take effect with lakefs cloud & enterprise version :((
#email:
#    smtp_host: smtp.gmail.com
#    smtp_port: 587
#    use_ssl: true
#    user_name: ${GMAIL_USERNAME}
#    password: ${GMAIL_PASSWORD}
#    local_name: local_name
#    sender: sender
#    limit_every_duration: 1m
#    burst: 3
#    lakefs_base_url: 0.0.0.0:8000

# Only take effect when deploying with local storage blockstore
#installation:
#    user_name: trong
#    access_key_id: AKIAIOSFOLKFSSAMPLES
#    secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
EOF
