# a docker file for an sample training / mining / infer executor

# FROM ubuntu:20.04
FROM python:3.8.16

ENV LANG=C.UTF-8

# Change mirror
RUN sed -i 's#http://archive.ubuntu.com#http://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#http://mirrors.ustc.edu.cn#g' /etc/apt/sources.list

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo 'Asia/Shanghai' >/etc/timezone

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
# copy user code to WORKDIR
COPY ./app/*.py /app/

# copy user config template and manifest.yaml to /img-man
RUN mkdir -p /img-man
COPY img-man/*.yaml /img-man/

# view https://github.com/protocolbuffers/protobuf/issues/10051 for detail
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically

RUN echo "python /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
