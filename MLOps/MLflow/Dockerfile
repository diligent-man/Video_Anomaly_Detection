FROM ghcr.io/mlflow/mlflow:v2.17.0

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y curl

# Dependency for psycopg2
RUN apt install -y libpq-dev python3-dev g++
RUN pip install psycopg2

COPY health_check.py /
COPY docker-entrypoint.sh /usr/local/bin

EXPOSE 5000
ENTRYPOINT ["docker-entrypoint.sh"]
