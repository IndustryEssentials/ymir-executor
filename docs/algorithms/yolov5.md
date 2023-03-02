# yolov5 代码库简介

## 安装

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## 训练

```
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

## 推理

```
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

## 数据集

参考[yolov5自定义数据集](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml)

- 数据集配置文件 `dataset.yaml`

yolov5通过读取yaml配置文件，获得数据集的以下信息：

    - path: 数据集的根目录

    - train: 训练集划分，可以是一个目录，也可以是一个索引文件，或者是一个列表

    - val: 验证集划分

    - test: 测试集划分

    - names: 数据集的类别信息

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
  0: person
  1: bicycle
  2: car
  ...
  77: teddy bear
  78: hair drier
  79: toothbrush

```

- 数据集划分索引文件

每行均为图像文件的路径，示例如下：
```
coco128/images/im0.jpg
coco128/images/im1.jpg
coco128/images/im2.jpg
```

- 标注文件

    - 标注文件的路径通过图像文件的路径进行替换得到，会将其中的 `/images/` 替换为 `/labels/`, 文件后辍替换为 `.txt`， 具体代码如下：

    ```
    def img2label_paths(img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    ```

    - 标注文件采用txt格式, 每行为一个标注框，采用 `class_id x_center y_center width height` 的格式， 以空格进行分割。

    ![](../imgs/yolov5_ann_format.jpg)

    - `class_id`: 表示标注框所属类别的整数，从0开始计数

    - `x_center`: 归一化后标注框的中心 x 坐标，浮点数，取值范围为[0, 1]

    - `y_center`: 归一化后标注框的中心 y 坐标，浮点数，取值范围为[0, 1]

    - `width`: 归一化后的标注框宽度，浮点数，取值范围为[0, 1]

    - `height`: 归一化后的标注框亮度，浮点数，取值范围为[0, 1]

    - 标注文件内容示例如下：

    ```
    0 0.48 0.63 0.69 0.71
    0 0.74 0.52 0.31 0.93
    4 0.36 0.79 0.07 0.40
    ```
