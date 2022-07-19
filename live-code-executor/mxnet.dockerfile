ARG CUDA="11.2.1"
ARG CUDNN="8"
ARG BUILD="runtime" # runtime/devel
ARG SYSTEM="ubuntu18.04"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${BUILD}-${SYSTEM}
ARG MXNET="1.9.1"
ARG OPENCV="4.1.2.30"
ARG NUMPY="1.20.0"
ARG DEBIAN_FRONTEND="noninteractive"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh"

ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
# install linux package, needs to fix GPG error first.
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y git gcc wget curl zip libglib2.0-0 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Install python package
# view https://mxnet.apache.org/versions/1.9.1/get_started for detail
RUN pip3 install mxnet-cu112==${MXNET} loguru opencv-python==${OPENCV} numpy==${NUMPY}
# install ymir-exc sdk
RUN if [ "${SERVER_MODE}" = "dev" ]; then \
        pip install "git+https://github.com/IndustryEssentials/ymir.git/@dev#egg=ymir-exc&subdirectory=docker_executor/sample_executor/ymir_exc"; \
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
