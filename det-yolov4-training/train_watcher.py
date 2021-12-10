import logging
import os

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import convert_model_darknet2mxnet_yolov4


class __DarknetToMxnetHandler(FileSystemEventHandler):
    def __init__(self, width: int, height: int, class_num: int) -> None:
        super().__init__()
        self._image_width = width
        self._image_height = height
        self._class_numbers = class_num

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path: str = event.src_path
        if not os.path.splitext(src_path)[1] != 'weights':
            return

        # if file *.weights modified, convert it to mxnet params
        convert_model_darknet2mxnet_yolov4.run(num_of_classes=self._class_numbers,
                                               input_h=self._image_height,
                                               input_w=self._image_width,
                                               load_param_name=src_path,
                                               export_dir='/out/models/model')


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

        self._observer = Observer()
        event_handler = __DarknetToMxnetHandler()
        self._observer.schedule(event_handler=event_handler, path=self._model_dir)

        self._observer.start()

    def stop(self) -> None:
        if not self._observer:
            return
        self._observer.stop()
