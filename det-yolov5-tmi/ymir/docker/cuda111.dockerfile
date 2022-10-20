ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="1.1.0"


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8
ENV YMIR_VERSION=$YMIR

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /workspace/
# install ymir-exc sdk and requirements
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.0.0" \
    && pip install -r /workspace/requirements.txt

# Copy file from host to docker and install requirements
COPY . /app
RUN mkdir /img-man && mv /app/*-template.yaml /img-man/

# Download pretrained weight and font file
RUN cd /app && bash data/scripts/download_weights.sh \
    && mkdir -p /root/.config/Ultralytics \
    && wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf

# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
