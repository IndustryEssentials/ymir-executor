import os

import yaml

from active_learning import DockerALAPI
from active_learning.utils import log_collector, LogWriter
import write_result


def _load_config() -> dict:
    with open("/in/config.yaml", "r", encoding='utf8') as f:
        config = yaml.safe_load(f)

    model_params_path = config["model_params_path"]
    base_dir = os.path.dirname(model_params_path)
    model_params_path = model_params_path if os.path.isabs(model_params_path) else os.path.join(
        base_dir, model_params_path)
    config["model_params_path"] = model_params_path

    if "task_id" not in config:
        config["task_id"] = "0"
    return config


if __name__ == '__main__':
    config = _load_config()

    log_writer = LogWriter(monitor_path="/out/monitor.txt",
                           monitor_pure_path="/out/monitor-log.txt",
                           summary_path="/out/log.txt")
    log_collector.set_logger(log_writer, config["task_id"], verbose=True)
    log_collector.monitor_collect(0.00, "pending", per_seconds=0)
    log_collector.summary_collect("config: {}".format(config))

    if 'write_result' not in config or not config["write_result"]:
        # mining
        api = DockerALAPI(candidate_path="/in/candidate/index.tsv", result_path="/out/result.tsv", **config)
        api.run()
    else:
        # infer (currently without mining)
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
