# yolov4 镜像说明文档

## 仓库地址

> 参考仓库 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [det-yolov4-tmi](https://github.com/modelai/ymir-executor-fork/tree/master/det-yolov4-tmi)

## 镜像地址
```
youdaoyzbx/ymir-executor:ymir2.0.0-yolov4-cu112-tmi
```

## 性能表现

> 参考文档 [yolov4 model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo)

| model | size | mAP@0.5:0.95 | mAP@0.5 |
| - | - | - | - |
| yolov4 | 608 | 43.5 | 65.7 |
| yolov4-Leaky | 608 | 42.9 | 65.3 |
| yolov4-Mish | 608 | 43.8 | 65.6 |

## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| image_height | 608 | 整数 | 输入网络的图像高度 | 采用 32的整数倍，如416, 512, 608 |
| image_width | 608 | 整数 | 输入网络的图像宽度 | 采用 32的整数倍，如416, 512, 608 |
| learning_rate | 0.0013 | 浮点数 | 学习率 | 采用默认值即可 |
| max_batches | 20000 | 整数 | 训练次数 | 如要减少训练时间，可减少max_batches |
| warmup_iterations | 1000 | 整数 | 预热训练次数 | 采用默认值即可 |
| batch | 64 | 整数 | 累计梯度的批处理大小，即batch size | 采用默认值即可 |
| subdivisions | 64 | 整数 | 累计梯度的次数 | 需要是batch参数的因数，如32。其中64表示一次加载一张图片，累计梯度64次；32表示一次加载两张图片，共累计32次。实际的batch size均为64。|

**说明**
1. 过于复杂的参数anchors不做说明，保持默认即可


## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| image_height | 608 | 整数 | 输入网络的图像高度 | 采用 32的整数倍，如416, 512, 608 |
| image_width | 608 | 整数 | 输入网络的图像宽度 | 采用 32的整数倍，如416, 512, 608 |
| confidence_thresh | 0.1 | 浮点数 | 置信度阈值 | - |
| nms_thresh | 0.45 | 浮点数 | nms时的iou阈值 | - |
| max_boxes | 50 | 整数 | 每张图像最多检测的目标数量 | - |

## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| data_workers | 28 | 整数 | 读取数据时使用的进程数量 | - |
| strategy | aldd_yolo | 字符串 | 挖掘算法 | - |
| image_height | 608 | 整数 | 输入网络的图像高度 | 采用 32的整数倍，如416, 512, 608 |
| image_width | 608 | 整数 | 输入网络的图像宽度 | 采用 32的整数倍，如416, 512, 608 |
| batch_size | 4 | 整数 | 批处理大小 | - |
| confidence_thresh | 0.1 | 浮点数 | 置信度阈值 | - |
| nms_thresh | 0.45 | 浮点数 | nms时的iou阈值 | - |
| max_boxes | 50 | 整数 | 每张图像最多检测的目标数量 | - |
