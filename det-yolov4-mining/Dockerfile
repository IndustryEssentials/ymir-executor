FROM industryessentials/mxnet_python:1.5.0_gpu_cu101mkl_py3_ub18

RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/* \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/* \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y --no-install-recommends libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel && pip3 install opencv-python pyyaml scipy tqdm && rm -rf /root/.cache/pip3

COPY . /app
WORKDIR /app
RUN cp ./start.sh /usr/bin/start.sh && \
    mkdir -p /img-man && \
    cp ./mining-template.yaml /img-man/mining-template.yaml && \
    cp ./infer-template.yaml /img-man/infer-template.yaml && \
    cp ./README.md /img-man/readme.md
CMD sh /usr/bin/start.sh
