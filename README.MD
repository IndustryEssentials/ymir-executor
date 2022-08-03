# ymir-executor documentation [English](./README.MD) | [简体中文](./README_zh-CN.MD)

- [ymir](https://github.com/IndustryEssentials/ymir)

## ymir-1.1.0 official image

- [yolov4](https://github.com/yzbx/ymir-executor-fork#det-yolov4-training)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu112-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu101-tmi
    ```

- [yolov5](https://github.com/yzbx/ymir-executor-fork#det-yolov5-tmi)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu102-tmi
    ```

- [mmdetection](https://github.com/yzbx/ymir-executor-fork#det-mmdetection-tmi)
    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu102-tmi
    ```

- [detectron2](https://github.com/yzbx/ymir-detectron2)

    - [change log](https://github.com/yzbx/ymir-detectron2/blob/master/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-detectron2-cu111-tmi
    ```

- [yolov7](https://github.com/yzbx/ymir-yolov7)

    - [change log](https://github.com/yzbx/ymir-yolov7/blob/main/ymir/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov7-cu111-tmi
    ```

## det-yolov4-tmi

- yolov4 training, mining and infer docker image, use `mxnet` and `darknet` framework

  ```
  cd det-yolov4-tmi
  docker build -t ymir-executor/yolov4:cuda101-tmi -f cuda101.dockerfile .

  docker build -t ymir-executor/yolov4:cuda112-tmi -f cuda112.dockerfile .
  ```

## det-yolov5-tmi

- [change log](./det-yolov5-tmi/README.md)

- yolov5 training, mining and infer docker image, use `pytorch` framework

```
cd det-yolov5-tmi
docker build -t ymir-executor/yolov5:cuda102-tmi -f cuda102.dockerfile .

docker build -t ymir-executor/yolov5:cuda111-tmi -f cuda111.dockerfile .
```

## det-mmdetection-tmi

- [change log](./det-mmdetection-tmi/README.md)

```
cd det-mmdetection-tmi
docker build -t ymir-executor/mmdet:cu102-tmi -f docker/Dockerfile.cuda102 .

docker build -t ymir-executor/mmdet:cu111-tmi -f docker/Dockerfile.cuda111 .
```

## live-code-executor

- 可以通过`git_url`, `commit id` 或 `tag` 从网上clone代码到镜像并运行, 不推荐使用`branch`, 因为这样拉取的代码可能随时间变化, 实验结果不具备可重复性.

- 参考 [live-code](https://github.com/IndustryEssentials/ymir-remote-git)

```
cd live-code-executor

docker build -t ymir-executor/live-code:torch-tmi -f torch.dockerfile

docker build -t ymir-executor/live-code:mxnet-tmi -f mxnet.dockerfile
```

## how to custom ymir-executor

- [custom ymir-executor](https://github.com/IndustryEssentials/ymir/blob/dev/docs/ymir-dataset-zh-CN.md)

- [ymir-executor-sdk](https://github.com/yzbx/ymir-executor-sdk)

## how to import pretrained model weights

- [import pretainted model weights](https://github.com/IndustryEssentials/ymir/blob/dev/docs/import-extra-models.md)
