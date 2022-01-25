from asyncio import tasks
import os
from typing import List

import yaml

from active_learning import DockerALAPI
from active_learning.utils import log_collector, LogWriter, TaskState
import write_result


def _load_config() -> dict:
    with open("/in/config.yaml", "r", encoding='utf8') as f:
        config = yaml.safe_load(f)

    # set default task id
    if "task_id" not in config:
        config["task_id"] = "0"

    # select mxnet model for mining and infer
    model_params_path_conf: List[str] = config['model_params_path']
    model_params_path = ''
    for p in model_params_path_conf:
        if os.path.splitext(p)[1] == '.params':
            model_params_path = p
            break

    if not model_params_path:
        raise ValueError(f"can not find mxnet params model in model_params_path: {model_params_path_conf}")

    config['model_params_path'] = model_params_path

    return config


if __name__ == '__main__':
    config = _load_config()

    log_writer = LogWriter(monitor_path="/out/monitor.txt",
                           monitor_pure_path="/out/monitor-log.txt",
                           summary_path="/out/log.txt")
    log_collector.set_logger(log_writer, config["task_id"], verbose=True)
    log_collector.monitor_collect(0.00, TaskState.PENDING, per_seconds=0)
    log_collector.summary_collect("config: {}".format(config))

    run_infer = int(config['run_infer'])
    run_mining = int(config['run_mining'])

    if not run_infer and not run_mining:
        raise ValueError('both run_infer and run_mining set to 0, abort')

    if run_mining:
        # mining
        print('>>> RUN MINING <<<')
        print(f"with model: {config['model_params_path']}")
        api = DockerALAPI(candidate_path="/in/candidate/index.tsv", result_path="/out/result.tsv", **config)
        api.run()
    if run_infer:
        # infer
        print('>>> RUN INFER <<<')
        print(f"with model: {config['model_params_path']}")
        gpu_id = config.get('gpu_id', '')
        confidence_thresh = float(config["confidence_thresh"])
        nms_thresh = float(config["nms_thresh"])
        image_height = int(config["image_height"])
        image_width = int(config["image_width"])
        model_params_path = config["model_params_path"]
        anchors = config["anchors"]
        class_names = config["class_names"]
        batch_size = config.get('batch_size', 1)

        write_result.run(candidate_path='/in/candidate/index.tsv',
                         result_path='/out/infer-result.json',
                         gpu_id=gpu_id,
                         confidence_thresh=confidence_thresh,
                         nms_thresh=nms_thresh,
                         image_width=image_width,
                         image_height=image_height,
                         model_params_path=model_params_path,
                         anchors=anchors,
                         class_names=class_names,
                         batch_size=batch_size)
