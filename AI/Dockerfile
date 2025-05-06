FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common curl
RUN apt update -y
RUN apt install -y ffmpeg

RUN apt-add-repository ppa:deadsnakes/ppa
RUN apt install -y python3-pip python3.11 python3.11-venv

RUN adduser nonroot
RUN mkdir /AI && chown -R nonroot:nonroot /AI
RUN mkdir /weight && chown -R nonroot:nonroot /weight

COPY --chown=nonroot:nonroot . /AI

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.11 -m venv $VIRTUAL_ENV
# RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /AI/requirements.txt

EXPOSE 6967
CMD ["python3.11", "/AI/main.py"]
