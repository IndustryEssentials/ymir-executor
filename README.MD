# ymir-executor documentation [English](./README.MD) | [简体中文](./README_zh-CN.MD)

- [ymir](https://github.com/IndustryEssentials/ymir)

- [wiki](https://github.com/modelai/ymir-executor-fork/wiki)

## ymir-1.1.0 official image

- [yolov4](https://github.com/modelai/ymir-executor-fork#det-yolov4-training)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu112-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu101-tmi
    ```

- [yolov5](https://github.com/modelai/ymir-executor-fork#det-yolov5-tmi)

    - [change log](./det-yolov5-tmi/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu102-tmi
    ```

- [mmdetection](https://github.com/modelai/ymir-executor-fork#det-mmdetection-tmi)

    - [change log](./det-mmdetection-tmi/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu102-tmi
    ```

- [detectron2](https://github.com/modelai/ymir-detectron2)

    - [change log](https://github.com/modelai/ymir-detectron2/blob/master/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-detectron2-cu111-tmi
    ```

- [yolov7](https://github.com/modelai/ymir-yolov7)

    - [change log](https://github.com/modelai/ymir-yolov7/blob/main/ymir/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov7-cu111-tmi
    ```

- [vidt](https://github.com/modelai/ymir-vidt)

    - [change log](https://github.com/modelai/ymir-vidt/tree/main/ymir)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-vidt-cu111-tmi
    ```

- [nanodet](https://github.com/modelai/ymir-nanodet/tree/ymir-dev)

    - [change log](https://github.com/modelai/ymir-nanodet/tree/ymir-dev/ymir)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-nanodet-cu111-tmi
    ```

## overview

| docker image | [finetune](https://github.com/modelai/ymir-executor-fork/wiki/use-yolov5-to-finetune-or-training-model) | tensorboard | args/cfg options | framework | onnx | pretrained weights |
| - | - | - | - | - | - | - |
| yolov4 | ? | ✔️ | ❌ | darknet + mxnet | ❌ | local |
| yolov5 | ✔️ | ✔️ | ✔️ | pytorch | ✔️ | local+online |
| yolov7 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | local+online |
| mmdetection | ✔️ | ✔️ | ✔️ | pytorch | ❌ | online |
| detectron2 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | online |
| vidt | ? | ✔️ | ✔️ | pytorch | ❌ | online |
| nanodet | ✔️ | ✔️ | ❌ | pytorch_lightning | ❌ | online |

- `online` pretrained weights may download through network

- `local` pretrained weights have copied to docker images when building image

### benchmark

- training dataset: voc2012-train 5717 images
- validation dataset: voc2012-val 5823 images
- image size: 640

gpu: single Tesla P4

| docker image | batch size | epoch number | model | voc2012 val map50 | training time | note |
| - | - | - | - | - | - | - |
| yolov5 | 16 | 100 | yolov5s | 70.05% | 9h | coco-pretrained |
| vidt | 2 | 100 | swin-nano | 54.13% | 2d | imagenet-pretrained |
| yolov4 | 4 | 20000 steps | yolov4 | 66.18% | 2d | imagenet-pretrained |
| yolov7 | 16 | 100 | yolov7-tiny | 70% | 8h | coco-pretrained |

gpu: single GeForce GTX 1080 Ti

| docker image | batch size | epoch number | model | voc2012 val map50 | training time | note |
| - | - | - | - | - | - | - |
| yolov5 | 16 | 100 | yolov5s | 70.35% | 2h | coco-pretrained |
| yolov7 | 16 | 100 | yolov7-tiny | 70.4% | 5h | coco-pretrained |
| mmdetection | 16 | 100 | yolox_tiny | 66.2% | 5h | coco-pretrained |
| detectron2 | 2 | 20000 steps | retinanet_R_50_FPN_1x | 53.54% | 2h | imagenet-pretrained |

---

## det-yolov4-tmi

- yolov4 training, mining and infer docker image, use `mxnet` and `darknet` framework

  ```
  cd det-yolov4-tmi
  docker build -t ymir-executor/yolov4:cuda101-tmi -f cuda101.dockerfile .

  docker build -t ymir-executor/yolov4:cuda112-tmi -f cuda112.dockerfile .
  ```

## det-yolov5-tmi

- yolov5 training, mining and infer docker image, use `pytorch` framework

```
cd det-yolov5-tmi
docker build -t ymir-executor/yolov5:cuda102-tmi -f cuda102.dockerfile .

docker build -t ymir-executor/yolov5:cuda111-tmi -f cuda111.dockerfile .
```

## det-mmdetection-tmi

```
cd det-mmdetection-tmi
docker build -t ymir-executor/mmdet:cu102-tmi -f docker/Dockerfile.cuda102 .

docker build -t ymir-executor/mmdet:cu111-tmi -f docker/Dockerfile.cuda111 .
```

## how to custom ymir-executor

- [custom ymir-executor](https://github.com/IndustryEssentials/ymir/blob/dev/docs/ymir-dataset-zh-CN.md)

- [ymir-executor-sdk](https://github.com/modelai/ymir-executor-sdk)

## how to import pretrained model weights

- [import pretainted model weights](https://github.com/IndustryEssentials/ymir/blob/dev/docs/import-extra-models.md)

## reference

- [mining algorithm: CALD](https://github.com/we1pingyu/CALD/)
- [yolov4](https://github.com/AlexeyAB/darknet)
- [yolov5](https://github.com/ultralytics/yolov5)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [yolov7](https://github.com/wongkinyiu/yolov7)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [vidt](https://github.com/naver-ai/vidt)
- [nanodet](https://github.com/RangiLyu/nanodet)
