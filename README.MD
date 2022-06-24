# ymir-executor 使用文档

## det-yolov4-training

- yolov4的训练镜像，采用mxnet与darknet框架，默认cuda版本为`10.1`，无法直接在高版本显卡如GTX3080/GTX3090上运行，需要修改dockerfile将cuda版本提升为11.1以上，并修改其它依赖。

## det-yolov4-mining

- yolov4挖掘与推理镜像，与det-yolov4-training对应

## det-yolov5-tmi

- yolov5训练、挖掘及推理镜像，训练时会从github上下载权重

- yolov5-FAQ

    - 权重下载出错：提前将权重下载好并复制到镜像中

## live-code-executor

- 可以通过`git_url`, `git_branch`从网上clone代码到镜像并运行

    - 参考[live-code](https://github.com/IndustryEssentials/ymir-remote-git)

## det-mmdetection-tmi

- mmdetection 训练、挖掘及推理镜像，目前还没开发完

# 如何制作自己的ymir-executor

- [ymir-executor 制作指南](https://github.com/IndustryEssentials/ymir/blob/dev/docs/ymir-dataset-zh-CN.md)

# FAQ

- apt 或 pip 安装慢或出错

    - 采用国内源，如在docker file 中添加如下命令

    ```
    RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

    RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
    ```

- docker build 的时候出错，找不到相应docker file或`COPY/ADD`时出错

    - 回到项目根目录或docker file对应根目录，确保docker file 中`COPY/ADD`的文件与文件夹能够访问，以yolov5为例.

    ```
    cd ymir-executor

    docker build -t ymir-executor/yolov5 . -f det-yolov5-tmi/cuda111.dockerfile
    ```

- 镜像运行完`/in`与`/out`目录中的文件被清理

    - ymir系统为节省空间，会在任务结束后删除其中不必要的文件，如果不想删除，可以在部署ymir时，修改文件`ymir/command/mir/tools/command_run_in_out.py`，注释其中的`_cleanup(work_dir=work_dir)`。注意需要重新构建后端镜像

    ```
    cd ymir
    docker build -t industryessentials/ymir-backend --build-arg PIP_SOURCE=https://pypi.mirrors.ustc.edu.cn/simple --build-arg SERVER_MODE='dev' -f Dockerfile.backend .

    docker-compose down -v && docker-compose up -d
    ```
