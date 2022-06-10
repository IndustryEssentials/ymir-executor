ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
ARG USER_GID=1000
ARG USER_UID=1000
ARG USER=ymir

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# install linux package
RUN apt-get update && apt-get install -y git curl wget zip gcc \
    libglib2.0-0 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python package
RUN pip install -U pip && \
    pip install loguru

# install ymir-exc sdk
RUN pip install ymir-exc

# copy template training/mining/infer config file
RUN mkdir -p /img-man
COPY img-man/*.yaml /img-man/
COPY start.sh /usr/bin

WORKDIR /workspace
COPY ymir_start.py /workspace/ymir_start.py

# set up python path
ENV PYTHONPATH=.

# Create non-root user and chown /workspace
RUN groupadd --gid $USER_GID $USER \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER --create-home \
    && chown ${USER_GID}:${USER_GID} /workspace

CMD bash /usr/bin/start.sh
