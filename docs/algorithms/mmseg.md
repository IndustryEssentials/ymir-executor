# ymir-mmsegmentation

## mmsegmentation简介

mmsegmentation 是 OpenMMLab 开源的语义分割工具库，包含众多的算法。可以阅读其[官方文档](https://mmsegmentation.readthedocs.io/zh_CN/latest/index.html)了解其详细用法，此处仅介绍其训练，推理相关的内容。

### 训练

- 单GPU训练的命令如下, 其中的 **CONFIG_FILE** 可以从 [configs](https://github.com/open-mmlab/mmsegmentation/tree/master/configs) 目录下找到

```
python tools/train.py ${CONFIG_FILE} [可选参数]

# 在cityscapes数据集上训练deeplabv3plus模型
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes.py
```

- 多GPU训练的命令如下
```
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [可选参数]

# 采用4块GPU进行训练
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes.py 4
```

### 推理

- 可以参考 [demo/image_demo.py](https://github.com/open-mmlab/mmsegmentation/tree/master/demo/image_demo.py)

- 先下载对应config的权重文件, 可在[configs/deeplabv3plus/README.md](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus/README.md)找到对应 **CONFIG_FILE** 和权重文件

```
wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth
```

- 进行推理
```
python demo/image_demo.py demo/demo.png configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth
```
