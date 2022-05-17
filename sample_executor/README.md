# ymir-1.0.0 基础训练/挖掘/推理镜像制作指南

* 参考 [ymir用户自定义镜像制作指南](https://github.com/IndustryEssentials/ymir/tree/master/docker_executor/sample_executor)

## 目的

* 规范化推理镜像接口，利用已有远程代码仓库快速生成镜像

## 接口

### 镜像基本要求

- 预装了 python 3.6 以上版本，git 及 pip，git 应有 global username & email 等初始配置

- 预装 mxnet 或 pytorch

- 预装 ymir-executor 框架 (即sample_executor)，此框架用于 app 与 ymir 交互，读取数据集，记录训练、推理、挖掘结果等

### 镜像启动时，会依次执行以下程序 

1. 读取/in/config.yaml，利用其中的git-url, git-branch 参数拉取代码到/app 目录

2. 读取/app/extra-requirements.txt，安装python依赖

3. 启动/app/start.py, start.py 可能需要下载一些预训练权重

### 代码仓库基本要求

- 提供extra-requiremnts.txt 与 start.py, start.py 可参考sample_executor/app/start.py

附加功能:

- 提供training-template.yaml, mining-template.yaml 与 infer-template.yaml 中的一个或多个，从而支持训练/挖掘/推理功能中的一个或多个

## 基础镜像构建

1. 下载 ymir-executor 工程：

```
git clone https://github.com/IndustryEssentials/ymir-executor
```

2. 根据服务器选择合适的镜像，下面以安装Nvidia cuda11.1 的服务器为例

注：服务器的cuda version需要>=容器的cuda version

```
docker build -t ymir-executor-python-base:cuda111 . -f sample_executor/cuda111.dockerfile
```

3. 安装 nvidia-docker 并启动容器

```
docker run -it --rm -v ~/.ssh:/root/.ssh -v xxx/in:/in -v xxx/out:/out ymir-executor-python-base:cuda111
```