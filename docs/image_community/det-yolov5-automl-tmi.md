# ymir-yolov5 automl 镜像说明文档

## 仓库地址

> 参考[ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [modelai/ymir-yolov5](https://github.com/modelai/ymir-yolov5/tree/ymir-automl)

## 镜像地址

```
youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu102-tmi
```

## 性能表现

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n]      |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s]      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m]      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l]      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6]     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6]     |1280 |44.8   |63.7   |385    |8.2    |3.6    |16.8   |12.6
|[YOLOv5m6]     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6]     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4

## 训练/推理/挖掘参数


| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| fast | true | 布尔型 | True表示要求速度快 | True, true, False, false 大写小均支持 |
| accurate | true | 布尔型 | True表示要求精度高 | True, true, False, false 大写小均支持 |
