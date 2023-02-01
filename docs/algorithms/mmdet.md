# mmdetection

此文档采用 `mmdetection v3.x` 架构，阅读此文档前，建议先了解[mmengine](https://mmengine.readthedocs.io/zh_CN/latest/get_started/introduction.html).

- [mmdetection v3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

- [ymir-mmdetection](https://github.com/modelai/ymir-mmdetection)

## 配置镜像环境

## 提供超参数模板文件与镜像配置文件

- [img-man/*-template.yaml](https://github.com/modelai/ymir-mmdetection/tree/ymir/ymir/img-man)

## 提供默认启动脚本

- [ymir/start.py](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/start.py)

- Dockerfile
```
RUN echo "python /app/ymir/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

### 训练

### 推理

### 挖掘

## 制作镜像 det/mmdet:tmi

- [ymir/Dockerfile](https://github.com/modelai/ymir-mmdetection/tree/ymir/ymir/Dockerfile)

```
docker build -t det/mmyolo:tmi -f ymir/Dockerfile .
```
