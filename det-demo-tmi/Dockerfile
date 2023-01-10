# a docker file for an sample training / mining / infer executor

FROM python:3.8.13-alpine

# Add bash
RUN apk add bash
# Required to build numpy wheel
RUN apk add g++ git

COPY requirements.txt ./
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
# copy user code to WORKDIR
COPY ./app/start.py /app/

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
