import logging
import os

import yaml

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import convert_model_darknet2mxnet_yolov4


class _DarknetToMxnetHandler(FileSystemEventHandler):
    def __init__(self, width: int, height: int, class_num: int) -> None:
        super().__init__()
        self._image_width = width
        self._image_height = height
        self._class_numbers = class_num

        # None means have no best yet
        # if already have best, only handles best.weights
        self._best_base_name = None

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path: str = event.src_path
        src_basename = os.path.basename(src_path)
        src_main, src_ext = os.path.splitext(src_basename)
        is_best = ('best' in src_main)

        # for test
        with open('/out/ymir-tmp.txt', 'a') as f:
            f.write("file ext: {}, full path: {}\n".format(src_ext, src_path))
        # for test ends

        if src_ext != '.weights':
            # for test
            with open('/out/ymir-tmp.txt', 'a') as f:
                f.write("src_ext {} mismatch, return\n".format(src_ext))
            # for test ends
            return
        if not os.path.isfile(src_path):
            # for test
            with open('/out/ymir-tmp.txt', 'a') as f:
                f.write("src_path {} not file, return\n".format(src_path))
            # for test ends
            return
        if (not is_best) and self._best_base_name:
            # if already have best converted, we should convert the best modifications
            # for test
            with open('/out/ymir-tmp.txt', 'a') as f:
                f.write("src_path {} not best and already have best: {}, return\n".format(
                    src_path, self._best_base_name))
            # for test ends
            return

        # for test
        with open('/out/ymir-tmp.txt', 'a') as f:
            f.write("ready to convert: {}\n".format(src_path))
        # for test ends

        # if file *.weights modified, convert it to mxnet params
        convert_model_darknet2mxnet_yolov4.run(num_of_classes=self._class_numbers,
                                               input_h=self._image_height,
                                               input_w=self._image_width,
                                               load_param_name=src_path,
                                               export_dir='/out/models/model')

        if is_best:
            self._best_base_name = src_basename

        self._write_result_yaml(result_yaml_path='/out/models/result.yaml',
                                weights_base_name=(self._best_base_name or src_basename))

    def _write_result_yaml(self, result_yaml_path: str, weights_base_name: str) -> None:
        if os.path.isfile(result_yaml_path):
            with open(result_yaml_path, 'r') as f:
                results = yaml.safe_load(f.read())
        else:
            results = {}

        results['model'] = ['model-symbol.json', 'model-0000.params', weights_base_name]

        # for test
        with open('/out/ymir-tmp.txt', 'a') as f:
            f.write("ready to write results: {}\n".format(results))
        # for test ends

        with open(result_yaml_path, 'w') as f:
            yaml.dump(results, f)


class TrainWatcher:
    def __init__(self, model_dir: str, width: int, height: int, class_num: int) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._observer = None
        self._image_width = width
        self._image_height = height
        self._class_numbers = class_num

    def start(self) -> None:
        if self._observer:
            logging.warning('watcher is already running')
            return

        print("watcher start on path: {}".format(self._model_dir))
        self._observer = Observer()
        event_handler = _DarknetToMxnetHandler(width=self._image_width,
                                               height=self._image_height,
                                               class_num=self._class_numbers)
        self._observer.schedule(event_handler=event_handler, path=self._model_dir)

        self._observer.start()

    def stop(self) -> None:
        if not self._observer:
            return
        self._observer.stop()
