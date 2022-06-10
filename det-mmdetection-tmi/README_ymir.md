# det-mmdetection-tmi

`mmdetection` framework for object `det`ection `t`raining/`m`ining/`i`nfer task

# changelog
- modify `mmdet/datasets/coco.py`, save the evaluation result to `os.environ.get('COCO_EVAL_TMP_FILE')` with json format
- modify `mmdet/core/evaluation/eval_hooks.py`, write training result file and monitor task process
- modify `mmdet/datasets/__init__.py` and add `mmdet/datasets/ymir.py`, add class `YmirDataset` to load YMIR dataset.
