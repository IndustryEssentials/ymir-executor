# 导入外部模型权值

## import extra model for yolov5 (ymir2.0.0)

- create a tar file with weight file `best.pt` and config file `ymir-info.yaml`

```
$ tar -cf yolov5_best.tar best.pt ymir-info.yaml
$ cat ymir-info.yaml
best_stage_name: best
executor_config:
  class_names:
  - dog
package_version: 2.0.0
stages:
  best:
    files:
    - best.pt
    mAP: 0.8349897782446034
    stage_name: best
    timestamp: 1669186346
task_context:
  executor: youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi
  mAP: 0.8349897782446034
  producer: ymir
  task_parameters: '{"keywords": ["dog"]}'
  type: 1
```

![图片](https://user-images.githubusercontent.com/5005182/184783723-1ce48603-1254-4ed9-90ba-c1dd8510dc79.png)
