FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common
RUN apt-add-repository ppa:deadsnakes/ppa
RUN apt install -y python3-pip
RUN apt install -y python3.11 python3.11-venv
RUN python3.11 -m venv venv
RUN venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN venv/bin/pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["/venv/bin/python3"]
