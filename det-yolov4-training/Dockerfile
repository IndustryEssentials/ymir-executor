FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
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
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mxnet-cu101==1.5.1 pyyaml numpy opencv-python
RUN apt-get install -y libopencv-dev
COPY . /darknet
RUN make
CMD bash make_train_test_darknet.sh

