FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    ffmpeg \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    build-essential \
    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
RUN apt update
RUN apt-get install libopenmpi-dev -y

COPY ./setup/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./setup/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# python
RUN pip install --upgrade pip
RUN pip install numpy==1.26.0
RUN pip install transformer-lens circuitsvis
RUN pip install scipy matplotlib pandas plotly seaborn
RUN pip install scikit-learn
RUN pip install wandb
RUN pip install stable-baselines



