FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y software-properties-common curl
RUN apt-add-repository ppa:deadsnakes/ppa
RUN apt install -y python3-pip python3.11 python3.11-venv

RUN adduser nonroot
RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir -p /var/log/app && chown -R nonroot:nonroot /var/log/app

COPY --chown=nonroot:nonroot . .

ENV VIRTUAL_ENV=/home/model/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.11 -m venv $VIRTUAL_ENV
# RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6968
CMD ["python3.11", "main.py"]
