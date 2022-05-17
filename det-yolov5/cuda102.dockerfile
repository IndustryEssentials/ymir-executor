ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

# docker build -t yolov5/yolov5:cuda102 . -f det-yolov5/cuda102.dockerfile
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
	# apt-get update && apt-get install -y gnupg2 && \
	# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
	apt-get update && apt-get install -y gnupg2 git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 libgl1-mesa-glx ffmpeg sudo openssh-server \
    libyaml-dev vim tmux tree curl wget zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -U pip && \             
	pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
	pip install cython xtcocotools jupyter onnx onnx-simplifier loguru \
	tensorboard==2.5.0 numba progress yacs pthflops imagesize pydantic pytest \
	scipy pydantic pyyaml imagesize opencv-python thop pandas seaborn

WORKDIR /app
ADD ./det-yolov5 /app
# RUN pip install -r requirements.txt 

# 如果在内网使用，需要提前下载好yolov5 v6.1的权重与字体Arial.tff到指定目录
# COPY yolov5*.pt /app/
### wget https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf
# RUN mkdir -p /root/.config/Ultralytics
# COPY ./training/yolov5/Arial.ttf /root/.config/Ultralytics/Arial.ttf

# make PYTHONPATH include mmdetection and executor
ENV PYTHONPATH=.

# tmi framework and your app
COPY sample_executor /sample_executor
RUN pip install -e /sample_executor/executor
RUN mkdir /img-man
COPY ./det-yolov5/*-template.yaml /img-man/

# dependencies: write other dependencies here (pytorch, mxnet, tensorboard-x, etc.)

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /app/ymir_start.py