#!/bin/bash
# Create backend, auth db for mlflow service

set -e

psql --set ON_ERROR_STOP=1\
     --username "$POSTGRES_USER"\
     --dbname "$POSTGRES_DB"\
<<-EOSQL
    CREATE DATABASE auth;
    CREATE DATABASE backend;

    GRANT ALL PRIVILEGES ON DATABASE auth TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE backend TO $POSTGRES_USER;

    CREATE DATABASE lakefs;
    GRANT ALL PRIVILEGES ON DATABASE lakefs TO $POSTGRES_USER;
EOSQL
echo "Init complete"