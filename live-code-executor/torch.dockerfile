ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# install linux package
RUN apt-get update && apt-get install -y gnupg2 git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 libgl1-mesa-glx ffmpeg sudo openssh-server \
    libyaml-dev vim tmux tree curl wget zip build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python package
RUN pip install -U pip && \             
    pip install cython xtcocotools onnx onnx-simplifier loguru \
    tensorboard==2.5.0 numba progress yacs pthflops imagesize \
    pydantic pytest scipy pyyaml opencv-python thop pandas seaborn

# install ymir-exc sdk
RUN pip install ymir-exc

# copy template training/mining/infer config file
RUN mkdir -p /img-man
COPY app/*.yaml /img-man/
COPY start.sh /usr/bin
COPY ymir_start.py /workspace/ymir_start.py

# set up python path
ENV PYTHONPATH=.

# set up git
RUN git config --global user.name "zhangsan" && \
    git config --global user.email "zhangsan@163.com" 

CMD /usr/bin/start.sh
