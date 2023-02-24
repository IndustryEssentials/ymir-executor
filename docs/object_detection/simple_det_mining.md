# 制作一个简单的目标检测挖掘镜像

参考[ymir镜像制作简介](../overview/ymir-executor.md)

## 镜像输入输出示例
```
.
├── in
│   ├── annotations
│   ├── assets
│   ├── candidate-index.tsv
│   ├── config.yaml
│   ├── env.yaml
│   └── models
└── out
    ├── monitor.txt
    └── result.tsv
```

## 工作目录

```
cd det-demo-tmi
```

## 提供超参数模型文件

镜像中包含**/img-man/mining-template.yaml** 表示镜像支持挖掘

- [img-man/mining-template.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/img-man/mining-template.yaml)

```yaml
{!det-demo-tmi/img-man/mining-template.yaml!}
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
`COPY img-man/*.yaml /img-man/` 在复制mining-template.yaml的同时，会将manifest.yaml复制到镜像中的**/img-man**目录

## 提供默认启动脚本

- Dockerfile
```
RUN echo "python /app/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

- [app/start.py](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/det-demo-tmi/app/start.py)

::: det-demo-tmi.app.start._run_mining
    handler: python
    options:
      show_root_heading: false
      show_source: true

## 写进度

```
# use `monitor.write_monitor_logger` to write log to console and write task process percent to monitor.txt
logging.info(f"assets count: {len(lines)}, valid: {valid_image_count}")
monitor.write_monitor_logger(percent=0.2)

# if task done, write 100% percent log
logging.info('mining done')
monitor.write_monitor_logger(percent=1.0)
```

## 写结果文件

```
rw.write_mining_result(mining_result=mining_result)
```

## 制作镜像 demo/det:mining

```dockerfile
{!det-demo-tmi/Dockerfile!}
```

```
docker build -t demo/det:mining -f Dockerfile .
```
