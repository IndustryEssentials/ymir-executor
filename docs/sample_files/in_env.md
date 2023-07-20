```
input:
  annotations_dir: /in/annotations  # 标注文件存储目录
  assets_dir: /in/assets  # 图像文件存储目录
  candidate_index_file: /in/candidate-index.tsv  # 推理或挖掘任务中的数据集索引文件
  config_file: /in/config.yaml  # 超参数文件
  models_dir: /in/models  # 预训练模型文件存储目录
  root_dir: /in  # 输入信息根目录
  training_index_file: /in/train-index.tsv  # 训练任务中的训练数据集索引文件
  val_index_file: /in/val-index.tsv  # 训练任务中的验证数据集索引文件
output:
  infer_result_file: /out/infer-result.json  # 推理任务结果文件
  mining_result_file: /out/result.tsv  # 挖掘任务结果文件
  models_dir: /out/models  # 训练任务权重文件输出目录
  monitor_file: /out/monitor.txt  # 进度记录文件
  root_dir: /out  # 输出信息根目录
  tensorboard_dir: /out/tensorboard  # 训练任务中tensorboard日志目录
  training_result_file: /out/models/result.yaml  # 训练任务的结果文件
run_infer: false  # 是否执行推理任务
run_mining: true  # 是否执行挖掘任务
run_training: false  # 是否执行训练任务
protocol_version: 1.0.0  # ymir平台镜像接口版本
task_id: t00000020000029d077c1662111056  # 任务id
```

!!! 注意
    /in/env.yaml 中的所有路径均为绝对路径
