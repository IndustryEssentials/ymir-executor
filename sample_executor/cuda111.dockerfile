ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# docker build -t mmcv/mmcv:gtx3090 . -f cuda111.dockerfile
# cuda11.3 + pytorch 1.10.0
# cuda11.1 + pytorch 1.9.0
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

# tmi framework and your app
# COPY app /app --> git clone git-url -b git-branch -o /app
RUN mkdir -p /img-man /app
COPY sample_executor/app/*-template.yaml /img-man/
COPY sample_executor /sample_executor
RUN pip install -e /sample_executor/executor

ENV PYTHONPATH=.

# RUN git config --global --user.name "" && \
#     git config --global --user.email "zhang3@163.com" 

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /sample_executor/ymir_start.py