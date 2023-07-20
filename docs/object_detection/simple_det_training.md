# 制作一个简单的目标检测训练镜像

参考[ymir镜像制作简介](../overview/ymir-executor.md), 通过加载 /in 目录下的数据集，超参数，任务信息，预训练权重， 在 /out 目录下产生模型权重，进度文件，训练日志。

## 镜像输入输出示例
```
.
├── in
│   ├── annotations [257 entries exceeds filelimit, not opening dir]
│   ├── assets -> /home/ymir/ymir/ymir-workplace/sandbox/0001/training_asset_cache
│   ├── config.yaml
│   ├── env.yaml
│   ├── models
│   ├── train-index.tsv
│   └── val-index.tsv
├── out
│   ├── models [29 entries exceeds filelimit, not opening dir]
│   ├── monitor.txt
│   ├── tensorboard -> /home/ymir/ymir/ymir-workplace/ymir-tensorboard-logs/0001/t00000010000028774b61663839849
│   └── ymir-executor-out.log
└── task_config.yaml
```

## 工作目录
```
cd det-demo-tmi
```

## 提供超参数模型文件

镜像中包含**/img-man/training-template.yaml** 表示镜像支持训练

- [img-man/training-template.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/img-man/training-template.yaml)

指明数据格式 **export_format** 为 **det-ark:raw**, 即目标检测标注格式，详情参考[Ymir镜像数据集格式](../overview/dataset-format.md)

```yaml
{!det-demo-tmi/img-man/training-template.yaml!}
```

- [Dockerfile](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/Dockerfile)

```
RUN mkdir -p /img-man  # 在镜像中生成/img-man目录
COPY img-man/*.yaml /img-man/  # 将主机中img-man目录下的所有yaml文件复制到镜像/img-man目录
```

## 提供镜像说明文件

**object_type** 为 2 表示镜像支持目标检测

- [img-man/manifest.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/img-man/manifest.yaml)
```
# 2 for object detection
"object_type": 2
```

- Dockerfile
`COPY img-man/*.yaml /img-man/` 在复制training-template.yaml的同时，会将manifest.yaml复制到镜像中的**/img-man**目录

## 提供默认启动脚本

- Dockerfile
```
RUN echo "python /app/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

- [app/start.py](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/app/start.py)

::: det-demo-tmi.app.start._run_training
    handler: python
    options:
      show_root_heading: false
      show_source: true

## 写进度

```
if idx % monitor_gap == 0:
    monitor.write_monitor_logger(percent=0.2 * idx / N)

monitor.write_monitor_logger(percent=0.2)

monitor.write_monitor_logger(percent=1.0)
```

## 写结果文件

```
# use `rw.write_model_stage` to save training result
rw.write_model_stage(stage_name='epoch10',
                     files=['epoch10.pt', 'config.py'],
                     evaluation_result=dict(mAP=random.random() / 2))

rw.write_model_stage(stage_name='epoch20',
                     files=['epoch20.pt', 'config.py'],
                     evaluation_result=dict(mAP=expected_mAP))
```

## 写tensorboard日志

```
write_tensorboard_log(cfg.ymir.output.tensorboard_dir)
```

## 制作镜像 demo/det:training

```dockerfile
{!det-demo-tmi/Dockerfile!}
```

```
docker build -t demo/det:training -f Dockerfile .
```
