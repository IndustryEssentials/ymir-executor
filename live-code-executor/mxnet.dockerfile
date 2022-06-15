ARG CUDA="11.2.0"
ARG CUDNN="8"
ARG BUILD="runtime" # runtime/devel
ARG SYSTEM="ubuntu20.04"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${BUILD}-${SYSTEM}
ARG MXNET="1.9.1"
ENV LANG=C.UTF-8

ARG SERVER_MODE=prod

# install linux package, needs to fix GPG error first.
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y git wget curl python3-dev gcc zip libglib2.0-0 libgl1-mesa-glx && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install python package
# view https://mxnet.apache.org/versions/1.9.1/get_started for detail
RUN pip3 install mxnet-cu112==${MXNET} loguru

# install ymir-exc sdk
RUN if [ "${SERVER_MODE}" = "dev" ]; then \
    pip install --force-reinstall -U "git+https://github.com/IndustryEssentials/ymir.git/@dev#egg=ymir-exc&subdirectory=docker_executor/sample_executor/ymir_exc"; \
  else \
    pip install ymir-exc; \
  fi

# copy template training/mining/infer config file
RUN mkdir -p /img-man
COPY img-man/*.yaml /img-man/
COPY start.sh /usr/bin

WORKDIR /workspace
COPY ymir_start.py /workspace/ymir_start.py

# set up python path
ENV PYTHONPATH=.

CMD bash /usr/bin/start.sh
