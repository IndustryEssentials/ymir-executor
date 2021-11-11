FROM mxnet/python:1.5.0_gpu_cu101mkl_py3_ub18

RUN apt-get update && apt-get install -y --no-install-recommends libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel && pip3 install opencv-python pyyaml scipy tqdm && rm -rf /root/.cache/pip3

COPY . /app
WORKDIR /app
RUN cp ./start.sh /usr/bin/start.sh && \
    mkdir -p /img-man && \
    cp ./mining-template.yaml /img-man/mining-config-template.yaml && \
    cp ./infer-template.yaml /img-man/infer-config-template.yaml && \
    cp ./README.md /img-man/readme.md
CMD sh /usr/bin/start.sh
