from active_learning.dataset import DataReader, ImageFolderDataset
from active_learning.strategy import ALDD, CALD, ALDD_YOLO
from active_learning.model_inference import CenterNet
from active_learning.model_inference import YoloNet
from active_learning.utils import softmax, log_collector, try_exception_log

import os
import sys
import traceback
import numpy as np
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from concurrent.futures import ThreadPoolExecutor
import tqdm


models = {
    "centernet": CenterNet,
    "yolo": YoloNet
}

selectors = {
    "aldd": ALDD,
    "cald": CALD,
    "aldd_yolo":ALDD_YOLO
}


class DockerALAPI:
    def __init__(
        self, task_id="0", strategy="aldd", model_type="detection", model_name="centernet",
        model_params_path=None, detection_confidence_thresh=0.1,
        gpu_id=None, data_workers=28, image_width=608, image_height=608, class_distribution_score=[1.0], class_names=None, 
        candidate_path=None, result_path=None, batch_size=16, **kwargs):
        self.task_id = task_id
        self.strategy = strategy
        self.model_type = model_type
        self.model_name = model_name
        self.class_distribution_score = np.array([1.0] * len(class_names))
        if len(class_names) > 1:
            self.class_distribution_score = self.class_distribution_score.reshape(len(class_names), 1)
        self.num_of_class = len(class_names)

        self.gpu_ids = gpu_id.split(',') if gpu_id else None
        self.ctx = [mx.gpu(int(each)) for each in self.gpu_ids] if self.gpu_ids else [mx.cpu()]
        self.data_workers = data_workers
        self.detection_confidence_thresh = detection_confidence_thresh
        self.model_params_path = model_params_path
        self.result_path = result_path
        with open(candidate_path, "r") as f:
            self.img_list = f.readlines()
            base_dir = os.path.dirname(candidate_path)
            self.img_list = [os.path.join(base_dir, x) if not os.path.isabs(x) else x for x in self.img_list]
            self.img_list = [each.strip() for each in self.img_list]
        self.path2score = []
        self.total_num = len(self.img_list)
        self.progress_count = 0.0
        self.batch_size = 1 if self.strategy == "cald" else batch_size
        self.is_done = False
        log_collector.monitor_collect(0.00, "1", per_seconds=0)
        log_collector.summary_collect("init api success")
        self.transform = transforms.Compose([transforms.Resize(size=(image_width, image_height)),
                                             transforms.ToTensor()])

    @property
    def progress(self):
        return min(self.progress_count, 1)

    @property
    def is_completed(self):
        return self.is_done

    def _run(self):
        model = models[self.model_name](weights_file=self.model_params_path, ctx=self.ctx, num_of_class=self.num_of_class, class_distribution_score=self.class_distribution_score)
        selector = selectors[self.strategy](model=model)
        log_collector.summary_collect("start run")
        img_dataset = ImageFolderDataset(self.img_list)
        datareader = gluon.data.DataLoader(img_dataset.transform_first(self.transform),
                batch_size = (self.batch_size * len(self.ctx)), shuffle = False, num_workers = self.data_workers)

        tmp_result_filename, ext = os.path.splitext(self.result_path)
        # this result file stores net forward scores with img names without sorting
        tmp_result_filename = "{}_tmp{}".format(tmp_result_filename, ext)

        if os.path.isfile(tmp_result_filename):
            os.system("rm {}".format(tmp_result_filename))
        count = 0
        output_file_handle = open(tmp_result_filename, 'w')
        executor = ThreadPoolExecutor(max_workers=len(self.ctx))

        for batch in tqdm.tqdm(datareader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0, even_split=False)
            img_index = batch[1].asnumpy().tolist()
            results = []
            for result in executor.map(selector.compute_score, data):
                results.append(result)
            scores = np.concatenate(results)
            pick_img_paths = [img_dataset.img_list[each_index] for each_index in img_index]
            for each_imgpath, each_score in zip(pick_img_paths, scores):
                self.path2score.append([each_imgpath, each_score])
                output_str = "{}\t{}\n".format(each_imgpath, each_score)
                output_file_handle.write(output_str)
            count += self.batch_size * len(self.ctx)
            self.progress_count = float(count) / self.total_num
            log_collector.monitor_collect(self.progress, "2")
        output_file_handle.close()

        self.path2score.sort(key=lambda x: x[1], reverse=True)
        with open(tmp_result_filename, "w") as f:
            path2score = ["\t".join(list(map(str, x))) + "\n" for x in self.path2score]
            f.writelines(path2score)
        assert os.path.isfile(tmp_result_filename)
        os.system("mv {} {}".format(tmp_result_filename, self.result_path))

        self.is_done = True
        log_collector.monitor_collect(self.progress, "2", force=True)

    def run(self):
        try:
            self._run()
            log_collector.summary_collect("done")
        except Exception:
            exctype, value, tb = sys.exc_info()
            text = "".join(traceback.format_exception(exctype, value, tb))
            log_collector.monitor_collect(self.progress, "4", force=True)
            text = "".join(log_collector.error_msg) + text
            log_collector.error_collect(text)
            log_collector.summary_collect(text)
