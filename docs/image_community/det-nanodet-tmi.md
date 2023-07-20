# ymir-nanodet 镜像说明文档

> Super fast and high accuracy lightweight anchor-free object detection model. Real-time on mobile devices.

## 代码仓库

> 参考[RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)
- [modelai/ymir-nanodet](https://github.com/modelai/ymir-nanodet)

## 镜像地址
```
youdaoyzbx/ymir-executor:ymir2.0.0-nanodet-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.2-nanodet-cu111-tmi
```

## 性能说明

> 参考[RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)

Model          |Resolution| mAP<sup>val<br>0.5:0.95 |CPU Latency<sup><br>(i7-8700) |ARM Latency<sup><br>(4xA76) | FLOPS      |   Params  | Model Size
:-------------:|:--------:|:-------:|:--------------------:|:--------------------:|:----------:|:---------:|:-------:
NanoDet-m      | 320*320 |   20.6   | **4.98ms**           | **10.23ms**          | **0.72G**  | **0.95M** | **1.8MB(FP16)** &#124; **980KB(INT8)**
**NanoDet-Plus-m** | 320*320 | **27.0** | **5.25ms**       | **11.97ms**          | **0.9G**   | **1.17M** | **2.3MB(FP16)** &#124; **1.2MB(INT8)**
**NanoDet-Plus-m** | 416*416 | **30.4** | **8.32ms**       | **19.77ms**          | **1.52G**  | **1.17M** | **2.3MB(FP16)** &#124; **1.2MB(INT8)**
**NanoDet-Plus-m-1.5x** | 320*320 | **29.9** | **7.21ms**  | **15.90ms**          | **1.75G**  | **2.44M** | **4.7MB(FP16)** &#124; **2.3MB(INT8)**
**NanoDet-Plus-m-1.5x** | 416*416 | **34.1** | **11.50ms** | **25.49ms**          | **2.97G**   | **2.44M** | **4.7MB(FP16)** &#124; **2.3MB(INT8)**
YOLOv3-Tiny    | 416*416 |   16.6   | -                    | 37.6ms               | 5.62G      | 8.86M     |   33.7MB
YOLOv4-Tiny    | 416*416 |   21.7   | -                    | 32.81ms              | 6.96G      | 6.06M     |   23.0MB
YOLOX-Nano     | 416*416 |   25.8   | -                    | 23.08ms              | 1.08G      | 0.91M     |   1.8MB(FP16)
YOLOv5-n       | 640*640 |   28.4   | -                    | 44.39ms              | 4.5G       | 1.9M      |   3.8MB(FP16)
FBNetV5        | 320*640 |   30.4   | -                    | -                    | 1.8G       | -         |   -
MobileDet      | 320*320 |   25.6   | -                    | -                    | 0.9G       | -         |   -

***Download pre-trained models and find more models in [Model Zoo](#model-zoo) or in [Release Files](https://github.com/RangiLyu/nanodet/releases)***


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| config_file | config/nanodet-plus-m_416.yml | 文件路径 | 配置文件路径 | 参考[config](https://github.com/modelai/ymir-nanodet/tree/ymir-dev/config) |
| epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| input_size | -1 | 整数 | 输入模型的图像分辨率 | -1表示采用config_file中定义的图像大小 |
| learning_rate | -1 | 浮点数 | 学习率 | -1表示采用config_file中定义的学习率
| resume | False | 布尔型 | 是否继续训练 | 设置为True可实现提前中断与继续训练功能 |
| load_from | '' | 文件路径 | 加载权重位置 | 设置后可加载指定位置的权重文件 |


## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| conf_thres | 0.35 | 浮点数 | 置信度阈值 | - |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |


## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| conf_thres | 0.35 | 浮点数 | 置信度阈值 | - |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |

**说明**
1. nanodet仅支持aldd挖掘算法


## 引用

```
@misc{=nanodet,
    title={NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model.},
    author={RangiLyu},
    howpublished = {\url{https://github.com/RangiLyu/nanodet}},
    year={2021}
}
```
