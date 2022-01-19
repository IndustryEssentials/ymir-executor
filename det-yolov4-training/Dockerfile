FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
WORKDIR /darknet
RUN apt-get update
RUN apt install -y software-properties-common wget
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y python3.7
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN python3 get-pip.py
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mxnet-cu101==1.5.1 numpy opencv-python
RUN pip3 install pyyaml watchdog tensorboardX six
RUN echo '\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse\n' >> /etc/apt/sources.list
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y libopencv-dev
COPY . /darknet
RUN mkdir /img-man && cp /darknet/training-template.yaml /img-man/training-template.yaml
RUN make -j
CMD bash make_train_test_darknet.sh
