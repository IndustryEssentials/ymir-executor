ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0  + cudnn8 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# Install linux package
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
	apt-get update && apt-get install -y gnupg2 git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 libgl1-mesa-glx ffmpeg sudo openssh-server \
    libyaml-dev vim tmux tree curl wget zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python package
RUN pip install -U pip && \
	pip install cython xtcocotools jupyter onnx onnx-simplifier loguru \
	tensorboard==2.5.0 numba progress yacs pthflops pytest \
	scipy pydantic pyyaml imagesize opencv-python thop pandas seaborn

# Install ymir-exc sdk
RUN pip install ymir-exc

# Copy file from host to docker
ADD ./det-yolov5-tmi /app
RUN mkdir /img-man && cp /app/*-template.yaml /img-man/

# Download pretrained weight and font file
RUN cd /app && bash data/scripts/download_weights.sh
RUN mkdir -p /root/.config/Ultralytics && \
    wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf

# setup PYTHONPATH to find local package
ENV PYTHONPATH=.

WORKDIR /app
CMD python /app/ymir_start.py