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

ADD ./det-yolov5-tmi /app
RUN mkdir /img-man && cp /app/*-template.yaml /img-man/
# RUN pip install -r requirements.txt 

RUN bash data/scripts/download_weights.sh
RUN mkdir -p /workspace/.git
RUN pip install -i https://mirrors.aliyun.com/pypi/simple ymir-exc
RUN mkdir -p /root/.config/Ultralytics && \
    wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf

# make PYTHONPATH include mmdetection and executor
ENV PYTHONPATH=.

WORKDIR /app
# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /app/ymir_start.py