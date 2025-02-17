FROM ghcr.io/mlflow/mlflow:v2.17.0
#FROM ghcr.io/mlflow/mlflow:v2.20.2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y curl

# Dependency for psycopg2
RUN apt install -y libpq-dev python3-dev g++
RUN pip install psycopg2

COPY health_check.py /
COPY basic_auth.ini /usr/local/lib/python3.10/site-packages/mlflow/server/auth/
COPY docker-entrypoint.sh .

RUN export MLFLOW_ADMIN_USERNAME=${MLFLOW_ADMIN_USERNAME}
RUN export MLFLOW_ADMIN_PASSWORD=${MLFLOW_ADMIN_PASSWORD}

EXPOSE 5000
CMD ["bash", "docker-entrypoint.sh"]
