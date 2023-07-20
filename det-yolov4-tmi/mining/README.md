# yolov4 active learning and inference

## active learning

### Parameters

```python
from active_learning import ALAPI


api = ALAPI(
    selected_img_list_path=None,  # 默认设置为"./temp/{}_{}_result.txt".format(strategy, task_id)
    unlabeled_img_list_path="unlabeled_data.txt",  # 未标注图片txt文件路径 unlabeled_data
    labeled_img_list_path="labeled_data.txt",  # 已标注图片txt文件路径 labeled_data
    dst_unlabeled_img_list_path=None,  # 生成下一轮unlabeled data路径 unlabeled_data - select_data
    dst_labeled_img_list_path=None,  # 生成下一轮labeled data路径 labeled_data + select_data
    strategy="random",  # active learning选择策略
    proportion=None,  # al从unlabeled data中选择数据的比例
    absolute_number=5000,  # al从unlabeled data中选择数据的绝对数量（proportion优先级高于absolute_number，同时设置时，优先使用proportion）
    model_type="detection",  # 模型类型，目前支持detection
    model_name="centernet",  # 模型名称，目前支持centernet
    model_params_path=None,  # 训练好的模型参数文件路径
    gpu_id='0',  # '0,1,2,3'可以指定4块GPU
    data_workers=32,  # 读取数据时使用的进程数量
    task_id="al"  # 起到标识作用
)

默认生成文件
"./temp/{}_{}_result.txt".format(strategy, task_id)  # 仅有选中的图片
"./temp/{}_{}_score.txt".format(self.strategy, self.task_id)

```

### 启动方式

```shell
python al_main.py --help
```

docker方式见 [docker_readme](docker_readme.md)

### 模型接口

`model.get_heatmap(imgs: list[numpy.array[H, W, c]]) -> mxnet.array[N, C, H, W]`

## Inference

### Parameters

```python
import write_result

write_result.run(candidate_path='/in/candidate-index.tsv',  # path to assets index file
                 result_path='/out/infer-result.json',  # path to output result file
                 gpu_id=gpu_id,  # gpus to run, if None or empty, runs on cpu
                 confidence_thresh=confidence_thresh,  # conf thresh
                 nms_thresh=nms_thresh,  # nms thresh
                 image_width=image_width,  # image width, must be the same with trained model
                 image_height=image_height,  # image height, must be the same with trained model and image_width
                 model_params_path=model_params_path,  # model params
                 anchors=anchors,  # anchors, must be the same with trained model
                 class_names=class_names)  # class names, must be the same with trained model
```
