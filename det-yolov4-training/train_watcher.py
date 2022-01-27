import logging
import os
import re
from typing import Callable, Tuple

from tensorboardX import SummaryWriter
import yaml

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import convert_model_darknet2mxnet_yolov4


class _DarknetTrainingHandler(FileSystemEventHandler):
    # life circle
    def __init__(self, width: int, height: int, class_num: int) -> None:
        super().__init__()
        self._image_width = width
        self._image_height = height
        self._class_numbers = class_num
        self._tensorboard_writer = SummaryWriter(log_dir='/out/tensorboard')

        self._pattern_and_handlers: Tuple[str, Callable] = [
            ('^.*best.weights$', _DarknetTrainingHandler._on_best_weights_modified),
            ('^.*train-log.yaml$', _DarknetTrainingHandler._on_train_log_yaml_modified)
        ]

    def stop(self) -> None:
        self._tensorboard_writer.close()

    # public: interit from FileSystemEventHandler
    def on_modified(self, event: FileSystemEvent) -> None:
        if not os.path.isfile(event.src_path):
            return
        src_path: str = event.src_path
        src_basename = os.path.basename(src_path)
        for pattern, handler in self._pattern_and_handlers:
            if re.match(pattern=pattern, string=src_basename):
                try:
                    handler(self, src_path)
                except BaseException:
                    logging.exception(msg=f"error occured in handler: {handler} and path: {src_path}")

    # protected: pattern handlers
    def _on_best_weights_modified(self, src_path: str) -> None:
        # if file *best.weights modified, convert it to mxnet params
        convert_model_darknet2mxnet_yolov4.run(num_of_classes=self._class_numbers,
                                               input_h=self._image_height,
                                               input_w=self._image_width,
                                               load_param_name=src_path,
                                               export_dir='/out/models')

        self._write_result_yaml(result_yaml_path='/out/models/result.yaml',
                                weights_base_name=os.path.basename(src_path))

    def _on_train_log_yaml_modified(self, src_path: str) -> None:
        # if file result.yaml changed, write tensorboard
        with open(src_path, 'r') as f:
            train_log_dict = yaml.safe_load(f.read())

        if not isinstance(train_log_dict, dict):
            return

        iteration = int(train_log_dict['iteration'])
        loss = float(train_log_dict['loss'])
        avg_loss = float(train_log_dict['avg_loss'])
        rate = float(train_log_dict['rate'])
        mean_ap = train_log_dict.get('map', None)
        class_aps = train_log_dict.get('aps', None)

        self._tensorboard_writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=iteration)
        self._tensorboard_writer.add_scalar(tag="train/avg_loss", scalar_value=avg_loss, global_step=iteration)
        self._tensorboard_writer.add_scalar(tag="train/rate", scalar_value=rate, global_step=iteration)
        if isinstance(mean_ap, float):
            self._tensorboard_writer.add_scalar(tag="train/mAP", scalar_value=mean_ap, global_step=iteration)
        if class_aps and isinstance(class_aps, dict):
            self._tensorboard_writer.add_scalars(main_tag='train/aps', tag_scalar_dict=class_aps, global_step=iteration)
        self._tensorboard_writer.flush()

    # protected: general
    def _write_result_yaml(self, result_yaml_path: str, weights_base_name: str) -> None:
        if os.path.isfile(result_yaml_path):
            with open(result_yaml_path, 'r') as f:
                results = yaml.safe_load(f.read())
        else:
            results = {}

        results['model'] = ['model-symbol.json', 'model-0000.params', weights_base_name]

        with open(result_yaml_path, 'w') as f:
            yaml.dump(results, f)


class TrainWatcher:
    def __init__(self, model_dir: str, width: int, height: int, class_num: int) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._observer = None
        self._event_handler = _DarknetTrainingHandler(width=width, height=height, class_num=class_num)

    def start(self) -> None:
        if self._observer:
            logging.warning('watcher is already running')
            return

        self._observer = Observer()
        self._observer.schedule(event_handler=self._event_handler, path=self._model_dir)

        self._observer.start()

    def stop(self) -> None:
        if not self._observer:
            return
        self._observer.stop()
        self._event_handler.stop()
