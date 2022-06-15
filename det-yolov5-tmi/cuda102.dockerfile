ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx curl wget zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy file from host to docker
ADD ./det-yolov5-tmi /app
RUN mkdir /img-man && mv /app/*-template.yaml /img-man/
RUN pip install ymir-exc && pip install -r /app/requirements.txt

# Download pretrained weight and font file
RUN cd /app && bash data/scripts/download_weights.sh
RUN mkdir -p /root/.config/Ultralytics && \
    wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf

# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
CMD python3 /app/start.py