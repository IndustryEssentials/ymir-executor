```
args_options: --exist-ok
batch_size_per_gpu: 16
class_names:
- dog
- cat
- person
epochs: 10
export_format: ark:raw
gpu_count: 4
gpu_id: '0,1,2,3'
img_size: 640
model: yolov5s
num_workers_per_gpu: 8
opset: 11
save_period: 10
shm_size: 32G
sync_bn: false
task_id: t000000100000208ac7a1664337925
```
