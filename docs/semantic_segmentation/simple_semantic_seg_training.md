# 制作一个简单的语义分割训练镜像

参考[ymir镜像制作简介](../overview/ymir-executor.md)

## 工作目录
```
cd seg-semantic-demo-tmi
```

## 提供超参数模型文件

镜像中包含**/img-man/training-template.yaml** 表示镜像支持训练

- [img-man/training-template.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/img-man/training-template.yaml)

指明数据格式 **export_format** 为 **seg-coco:raw**, 即语义/实例分割标注格式，详情参考[Ymir镜像数据集格式](../overview/dataset-format.md)

```yaml
{!seg-semantic-demo-tmi/img-man/training-template.yaml!}
```

- [Dockerfile](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/Dockerfile)

```
RUN mkdir -p /img-man  # 在镜像中生成/img-man目录
COPY img-man/*.yaml /img-man/  # 将主机中img-man目录下的所有yaml文件复制到镜像/img-man目录
```

## 提供镜像说明文件

**object_type** 为 3 表示镜像支持语义分割

- [img-man/manifest.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/img-man/manifest.yaml)
```
# 3 for semantic segmentation
"object_type": 3
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

- [app/start.py](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/app/start.py)

::: seg-semantic-demo-tmi.app.start._run_training
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
rw.write_model_stage(stage_name='epoch20',
                     files=['epoch20.pt', 'config.py'],
                     evaluation_result=dict(mIoU=expected_miou))
```

## 写tensorboard日志

```
write_tensorboard_log(cfg.ymir.output.tensorboard_dir)
```

## 制作镜像 demo/semantic_seg:training

```dockerfile
{!seg-semantic-demo-tmi/Dockerfile!}
```

```
docker build -t demo/semantic_seg:training -f Dockerfile .
```
