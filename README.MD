# ymir-executor 使用文档

## det-yolov4-training

- yolov4的训练镜像，采用mxnet与darknet框架，默认cuda版本为`10.1`，无法直接在高版本显卡如GTX3080/GTX3090上运行，需要修改dockerfile将cuda版本提升为11.1以上，并修改其它依赖。

## det-yolov4-mining

- yolov4挖掘与推理镜像，与det-yolov4-training对应

## det-yolov5-tmi

- yolov5训练、挖掘及推理镜像，训练时会从github上下载权重

- yolov5-FAQ

    - 权重下载出错：提前将权重下载好并复制到镜像中或导入预训练模型

## live-code-executor

- 可以通过`git_url`, `git_branch`从网上clone代码到镜像并运行

- 参考 [live-code](https://github.com/IndustryEssentials/ymir-remote-git)

## det-mmdetection-tmi

- mmdetection 训练、挖掘及推理镜像，目前还没开发完


## 如何制作自己的ymir-executor

- [ymir-executor 制作指南](https://github.com/IndustryEssentials/ymir/blob/dev/docs/ymir-dataset-zh-CN.md)

## 如何导入预训练模型

- [如何导入外部模型](https://github.com/IndustryEssentials/ymir/blob/dev/docs/import-extra-models.md)

    - 通过ymir网页端的 `模型管理/模型列表/导入模型` 同样可以导入模型

---

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

    - ymir系统为节省空间，会在任务`成功结束`后删除其中不必要的文件，如果不想删除，可以在部署ymir时，修改文件`ymir/command/mir/tools/command_run_in_out.py`，注释其中的`_cleanup(work_dir=work_dir)`。注意需要重新构建后端镜像

        ```
        cd ymir
        docker build -t industryessentials/ymir-backend --build-arg PIP_SOURCE=https://pypi.mirrors.ustc.edu.cn/simple --build-arg SERVER_MODE='dev' -f Dockerfile.backend .

        docker-compose down -v && docker-compose up -d
        ```

- 训练镜像如何调试

    - 先通过失败任务的tensorboard链接拿到任务id，如`t000000100000175245d1656933456`

    - 进入ymir部署目录 `ymir-workplace/sandbox/work_dir/TaskTypeTraining/t000000100000175245d1656933456/sub_task/t000000100000175245d1656933456`, `ls` 可以看到以下结果

        ```
        # ls
        in  out  task_config.yaml
        ```

    - 挂载目录并运行镜像`<docker-image-name>`，注意需要将ymir部署目录挂载到镜像中

        ```
        docker run -it --gpus all -v $PWD/in:/in -v $PWD/out:/out -v <YMIR_WORKPLACE>:<YMIR_WORKPLACE> <docker-image-name> bash

        # 以/home/ymir/ymir-workplace作为ymir部署目录为例
        docker run -it --gpus all -v $PWD/in:/in -v $PWD/out:/out -v /home/ymir/ymir-workplace:/home/ymir/ymir-workplace <docker-image-name> bash
        ```

    - 推理与挖掘镜像调试同理，注意对应目录均为`ymir-workplace/sandbox/work_dir/TaskTypeMining`

- 模型精度/速度如何权衡与提升

    - 模型精度与数据集大小、数据集质量、学习率、batch size、 迭代次数、模型结构、数据增强方式、损失函数等相关，在此不做展开，详情参考：

        - [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)

        - [Paper with Code: Object Detection](https://paperswithcode.com/task/object-detection)

        - [awesome object detection](https://github.com/amusi/awesome-object-detection)
