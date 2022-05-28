ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# docker build -t ymir/yolov5:cuda111 -f det-yolov5-tmi/cuda111.dockerfile .
# cuda11.3 + pytorch 1.10.0
# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# Install linux package
# RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN	apt-get update && apt-get install -y gnupg2 git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 libgl1-mesa-glx ffmpeg sudo openssh-server \
    libyaml-dev vim tmux tree curl wget zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python package
# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install -U pip && \
	pip install cython xtcocotools onnx onnx-simplifier loguru \
	tensorboard==2.5.0 numba progress yacs pthflops imagesize pydantic pytest \
	scipy pyyaml opencv-python thop pandas seaborn

# Install ymir-exc sdk
RUN pip install ymir-exc

# Copy file from host to docker
ADD ./det-yolov5-tmi /app
RUN mkdir /img-man && cp /app/*-template.yaml /img-man/

# Download pretrained weight and font file
RUN cd /app && bash data/scripts/download_weights.sh
RUN mkdir -p /root/.config/Ultralytics && \
    wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf

# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
CMD python /app/ymir_start.py