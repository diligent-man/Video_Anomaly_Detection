<img src="https://img.shields.io/badge/Language-English-0011FF.svg"> | <img src="https://img.shields.io/badge/License-MIT-1047FE.svg"> | <img src="https://img.shields.io/badge/python-3.11-207CFD.svg"> | <img src="https://img.shields.io/badge/OS-linux, win-30B2FB.svg"> | <img src="https://img.shields.io/badge/Version-1.0.1-40E7FA.svg">

# Introduction
This source code implementation aims to assist researcher, user in creating DL-based algorithm for VAD problem quickly and systematically. In this project, we have:
* preprocessing pipeline script
* YouTube video crawling
* VAD train test splitter
* training script
* testing script
* mlflow export & import  

You can find all of them in [here](./src/tools) and associated running script in [here](./script). At the moment, we just implemented two algorithms due to time and resource constraints. The former is referenced from [Real-world Anomaly Detection in Surveillance Videos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf) and the second one is from [Distilling Aggregated Knowledge for Weakly-Supervised Video Anomaly
Detection](https://openaccess.thecvf.com/content/WACV2025/papers/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.pdf).

# Installation
## Prerequisites
#### 1/ NVIDIA CUDA on Linux (Optional)
check at [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

#### 2/ Docker (Required)
On Linux (Ubuntu): check at [here](https://docs.docker.com/engine/install/ubuntu/)  
On Windows: check at [here](https://viblo.asia/p/cai-dat-docker-tren-windows-10-3Q75w6gelWb)

#### 3/ NVIDIA Container Toolkit (Optional)
check at [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### 3/ FFmpeg
Only cpu:  
```bash
# Ubuntu 24.04
sudo apt update
sudo apt install ffmpeg
```  
With GPU support (tested on Ubuntu 24.04):  
```bash
# Install ffnvcodec
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && sudo make install && cd ~

# Install necessary packages
sudo apt-get install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev

# Install ffmpeg
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ && cd ffmpeg
./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
make -j 8
sudo make install
```
#### 4/ Swap memory (Linux only)
In case of insufficient memory during training or testing phases, we suggest that you should monitor your RAM and swap memory and have a suitable increase based on your need. For increasing swap memory on Ubuntu, check at [here](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04) 

#### 5/ Shared memory - shm (Linux only)
Due to multiprocessing mechanism of Pytorch DataLoader class, it requires a vast amount of shared memory for loading a torch-tensor-converted video.
```bash
# Add this line to the end of /etc/fstab file.
# This line increase shm capacity up to 80Gb
tmpfs /dev/shm tmpfs defaults,size=80G 0 0
```

#### 6/ Virtual memory (Windows only)
As above-mentioned reason, you should enlarge your virtual memory in case of using Windows OS. For more specific details, please check at [here](https://www.windowscentral.com/software-apps/windows-11/how-to-manage-virtual-memory-on-windows-11).

#### 7/ Python virtual environment
In order to evade dependencies conflict with other environment, we highly recommend you to create a completely new virtual python environment via [venv](https://docs.python.org/3/library/venv.html).

## Dependencies
#### Via pip
```bash
pip install -r requirement.txt
```
# Usage
### 1/ For crawling data from YouTube
Please read [this](./docs/en/crawl_data.md).
### 2/ For data preprocessing
Please read [this](./docs/en/data_preprocessing.md).
### 3/ For training model
Please read [this](./docs/en/cfg.md).
### 3/ For testing model
Please read at [here](./docs/en/cfg.md)
### 4/ For deploying model
Please read [this](./docs/en/deploy_model.md).
