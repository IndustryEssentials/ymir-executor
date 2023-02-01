# mmyolo

阅读此文档前，建议先了解[mmdet](./mmdet.md)

- [mmyolo](https://github.com/open-mmlab/mmyolo)

- [ymir-mmyolo](https://github.com/modelai/ymir-mmyolo)

## 配置镜像环境

参考 [mmyolo#installation](https://github.com/modelai/ymir-mmyolo#%EF%B8%8F-installation-)

- [ymir/Dockerfile](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/Dockerfile)

## 提供超参数模板文件

- [img-man/*-template.yaml](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/img-man)

## 提供镜像说明文件

- [img-man/manifest.yaml](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/img-man/manifest.yaml)

## 提供默认启动脚本

- [ymir/start.py](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/start.py)

- Dockerfile
```
RUN echo "python /app/ymir/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

完整代码变动参考[ymir-mmyolo/pull/1](https://github.com/modelai/ymir-mmyolo/pull/1/files)

### 训练

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_training.py`

4. `ymir_training.py` 调用 `bash tools/dist_train.sh ...`

    - `ymir_training.py` 调用 `convert_ymir_to_coco()` 实现数据集格式转换

    - `ymir_training.py` 获取配置文件(config_file)、GPU数量(num_gpus)、工作目录(work_dir)， 并拼接到调用命令中

    ```
    cmd = f"bash ./tools/dist_train.sh {config_file} {num_gpus} --work-dir {work_dir}"
    ```
    - 在训练结束后， 保存 `max_keep_checkpoints` 份权重文件

5. `dist_train.sh` 调用 `python3 tools/train.py ...`

    - `train.py` 中调用 `modify_mmengine_config()` 加载ymir平台超参数、自动配置预训练模型、添加tensorboard功能、添加ymir进度监控hook等。

### 推理

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_infer.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `mmdet_result_to_ymir()` 将mmdet推理结果转换为ymir格式

    - 调用 `rw.write_infer_result()` 保存推理结果

### 挖掘

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_mining.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `compute_score()` 计算挖掘分数

    - 调用 `rw.write_mining_result()` 保存挖掘结果

## 制作镜像 det/mmyolo:tmi

- [ymir/Dockerfile](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/Dockerfile)

```
docker build -t det/mmyolo:tmi -f ymir/Dockerfile .
```
