# ymir-yolov7 镜像说明文档

## 代码仓库

> 参考[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [modelai/ymir-yolov7](https://github.com/modelai/ymir-yolov7)

## 镜像地址

```
youdaoyzbx/ymir-executor:ymir2.1.0-yolov7-cu111-tmi
```

## 性能表现

> 数据参考[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| model | yolov5s | 字符串 | yolov5模型，可选yolov5n, yolov5s, yolov5m, yolov5l等 | 建议：速度快选yolov5n, 精度高选yolov5l, yolov5x, 平衡选yolov5s或yolov5m |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| img_size | 640 | 整数 | 输入模型的图像分辨率 | - |
| args_options | '--exist-ok' | 字符串 | yolov5命令行参数 | 建议：专业用户可用yolov5所有命令行参数 |
| save_weight_file_num | 1 | 整数 | 保存最新模型的数量 | - |
| sync_bn | False | 布尔型 | 是否同步各gpu上的归一化层 | 建议：开启以提高训练稳定性及精度 |
| cfg_file | cfg/training/yolov7-tiny.yaml | 文件路径 | 模型文件路径, 对应 `--cfg` | 参考[cfg/training](https://github.com/modelai/ymir-yolov7/tree/ymir/cfg/training) |
| hyp_file | data/hyp.scratch.tiny.yaml | 文件路径 | 超参数文件路径，对应 `--hyp` | 参考[data](https://github.com/modelai/ymir-yolov7/tree/ymir/data) |
| cache_images | True | 布尔 | 是否缓存图像 | 设置为True可加快训练速度 |


## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |

## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |

## 引用
```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
