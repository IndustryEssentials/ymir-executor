ARG CUDA="11.2.0"
ARG CUDNN="8"
ARG BUILD="runtime" # runtime/devel
ARG SYSTEM="ubuntu20.04"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${BUILD}-${SYSTEM}
ARG USER_GID=1000
ARG USER_UID=1000
ARG USER=ymir
ARG MXNET="1.9.1"
ENV LANG=C.UTF-8

# install linux package, needs to fix GPG error first.
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y git wget curl python3-dev gcc zip libglib2.0-0 libgl1-mesa-glx && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install python package
# view https://mxnet.apache.org/versions/1.9.1/get_started for detail
RUN pip3 install mxnet-cu112==${MXNET} loguru ymir-exc

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
