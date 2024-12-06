FROM ghcr.io/mlflow/mlflow:v2.18.0

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install -y nano

EXPOSE 5000
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "4"]
