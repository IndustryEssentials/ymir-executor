# ymir-1.0.0 远端基础训练/挖掘/推理镜像制作指南

* 参考 [ymir用户自定义镜像制作指南](https://github.com/IndustryEssentials/ymir/tree/master/docker_executor/sample_executor)

## 目的

* 规范化推理镜像接口，利用已有远程代码仓库快速生成镜像

## 接口

### 远端基础镜像基本要求

- 预装了 python 3.6 以上版本，git 及 pip，git 应有 global username & email 等初始配置

- 预装 mxnet 或 pytorch

- 预装 ymir-executor 框架 (即executor)，此框架用于 app 与 ymir 交互，读取数据集，记录训练、推理、挖掘结果, 报告训练进度等等。

### 镜像启动时，会依次执行以下程序 

1. 读取/in/config.yaml，利用其中的git-url, git-branch 参数，直接拉取代码到/app 目录

    - 注意代码的根目录在/app, 而不是/app/repo-name

    ```
    git pull git_url -b git_branch /app
    ```

2. 读取/app/extra-requirements.txt，安装python依赖

    - 默认为国内清华源, 可在/in/config.yaml 中设置pypi_mirror参数更改镜像源

    ```
    pypi_mirror=executor_config.get('pypi_mirror','https://pypi.tuna.tsinghua.edu.cn/simple')
    if pypi_mirror == '':
        cmd='pip install -r /app/extra-requirements.txt'
    else:
        # change to CN mirror to speed up "pip install".
        cmd=f'pip install -r /app/extra-requirements.txt -i {pypi_mirror}'
    ```

3. 在`/app` 目录下启动 `/app/start.py`

    - 注意如果镜像需要离线预训练权重，可以在/in/config.yaml中通过 pretrained_model_paths 参数指定，参考`app/training-template.yaml`

### 远端代码仓库要求

- [必选] start.py, 参考 `app/start.py`

- [可选] extra-requirements.txt, 与 pip requirements.txt 同格式

- [可选] configs/xxx.yaml, 代码配置文件，可以有多个，运行远端镜像中可通过 `code_config` 参数指定

## 远端基础镜像构建

1. 安装 ymir-exc：

    - 源码地址：https://github.com/IndustryEssentials/ymir/tree/dev/docker_executor/sample_executor/ymir_exc

    ```
    pip install ymir-exc -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

2. 根据服务器选择合适的镜像，下面以安装Nvidia cuda11.1 的服务器为例

    - 注：服务器的cuda version需要>=容器的cuda version

    ```
    docker build -t ymir-exc-base:torch1.8.0-cu111 . -f executor/torch.dockerfile \
        --build-arg PYTORCH=1.8.0 --build-arg CUDA=11.1
    ```

3. 安装 nvidia-docker 并启动容器

    ```
    docker run -it --rm -v xxx/in:/in -v xxx/out:/out ymir-exc-base:torch1.8.0-cu111
    ```