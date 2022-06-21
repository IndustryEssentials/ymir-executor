# det-mmdetection-tmi

`mmdetection` framework for object `det`ection `t`raining/`m`ining/`i`nfer task

# changelog
- modify `mmdet/datasets/coco.py`, save the evaluation result to `os.environ.get('COCO_EVAL_TMP_FILE')` with json format
- modify `mmdet/core/evaluation/eval_hooks.py`, write training result file and monitor task process
- modify `mmdet/datasets/__init__.py` and add `mmdet/datasets/ymir.py`, add class `YmirDataset` to load YMIR dataset.
- modify `mmdet/apis/train.py`, set `eval_cfg['classwise'] = True` for class-wise evaluation
- add `mmdet/utils/util_ymir.py` for ymir training/infer/mining
- add `ymir_infer.py` for infer and mining
- add `ymir_train.py` modify `tools/train.py` to update the mmcv config for training
