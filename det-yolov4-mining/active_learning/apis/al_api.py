from active_learning.dataset import LabeledDataset, UnlabeledDataset, DataReader
from active_learning.strategy import random_select, ALDD, CALD
from active_learning.model_inference import CenterNet

import os
import multiprocessing


models = {
    "centernet": CenterNet
}

selectors = {
    "aldd": ALDD,
    "cald": CALD
}


class ALAPI:
    def __init__(
        self, selected_img_list_path=None, unlabeled_img_list_path=None, labeled_img_list_path=None,
        dst_unlabeled_img_list_path=None, dst_labeled_img_list_path=None,
        strategy="random", proportion=None, absolute_number=None,
        model_type="detection", model_name="centernet", model_params_path=None, gpu_id='0', task_id="al",
        data_workers=28
    ):
        self.selected_img_list_path = selected_img_list_path
        self.unlabeled_img_list_path = unlabeled_img_list_path
        self.labeled_img_list_path = labeled_img_list_path
        self.strategy = strategy
        self.proportion = proportion
        self.absolute_number = absolute_number
        self.model_type = model_type
        self.model_name = model_name
        self.model_params_path = model_params_path
        self.gpu_ids = gpu_id.split(',')
        self.task_id = task_id
        self.data_workers = data_workers

        self.unlabeled_dataset = UnlabeledDataset(unlabeled_img_list_path, dst_unlabeled_img_list_path)
        self.labeled_dataset = LabeledDataset(labeled_img_list_path, dst_labeled_img_list_path)
        # self.datareader = DataReader(self.unlabeled_dataset.img_list, num_workers=data_workers)
        self.selected_img_list = []

        # compute sampled_num
        self.total_num = self.unlabeled_dataset.num_samples
        sampled_num = self.total_num
        if proportion:
            sampled_num = int(self.total_num * proportion)
        elif absolute_number:
            sampled_num = absolute_number
        self.sampled_num = min(sampled_num, self.total_num)

        self.path2score = []
        self.queue = multiprocessing.Queue(maxsize=512)
        self.progress_count = 0.0
        self.batch_size = 1 if self.strategy == "cald" else 16
        self.expand_ratio = 1.2
        self.is_done = False

    @property
    def progress(self):
        if self.strategy == "random":
            return 1.0
        else:
            return self.progress_count

    @property
    def is_completed(self):
        return self.is_done

    def random_select(self):
        #  random select
        selected_img_list = random_select(self.unlabeled_dataset, self.sampled_num)
        return selected_img_list

    def rank_select(self):
        return [x[0] + "\n" for x in self.path2score][:self.sampled_num]

    def cascade_rank_select(self):
        path2score = self.path2score[:int(self.sampled_num * self.expand_ratio)]
        path2score.sort(key=lambda x: x[2], reverse=True)
        return [x[0] + "\n" for x in path2score][:self.sampled_num]

    def select(self):
        if self.strategy == "random":
            selected_img_list = self.random_select()
        elif self.strategy == "aldd":
            selected_img_list = self.rank_select()
        elif self.strategy == "cald":
            selected_img_list = self.cascade_rank_select()
        return selected_img_list

    def sub_process_run(self, gpu_id, datareader):
        model = models[self.model_name](weights_file=self.model_params_path, gpu_id=gpu_id, confidence_thresh=0.1)
        selector = selectors[self.strategy](model=model, labeled_dataset=self.labeled_dataset)
        img_path_list = []
        imgs = []
        while True:
            img, img_path, stop = datareader.dequeue()
            img_path_list.append(img_path)
            imgs.append(img)
            if len(img_path_list) == self.batch_size or stop:
                scores = selector.compute_score(imgs)
                for img_path, score in zip(img_path_list, scores):
                    if isinstance(score, list):
                        item = [img_path]
                        item.extend(score)
                    else:
                        item = [img_path, score]
                    self.queue.put(item, block=True, timeout=None)
                img_path_list = []
                imgs = []
            if stop:
                break

    def run(self):
        if self.strategy == "random":
            self.selected_img_list = self.select()
            with open(self.selected_img_list_path, 'w') as f:
                f.writelines(self.selected_img_list)
            self.is_done = True
            return

        proc_pool = []
        cut_num = int(len(self.unlabeled_dataset.img_list) / len(self.gpu_ids))
        img_list_cut = [self.unlabeled_dataset.img_list[i * cut_num: (i + 1) * cut_num] for i in range(len(self.gpu_ids) - 1)]
        img_list_cut.append(self.unlabeled_dataset.img_list[(len(self.gpu_ids) - 1) * cut_num:])
        for i, gpu_id in enumerate(self.gpu_ids):
            img_list = img_list_cut[i]
            datareader = DataReader(img_list, num_workers=int(self.data_workers / len(self.gpu_ids)))
            datareader.start()
            proc = multiprocessing.Process(target=self.sub_process_run, args=(gpu_id, datareader))
            proc.daemon = True
            proc_pool.append(proc)
            proc.start()

        count = 0
        while True:
            result = self.queue.get(block=True, timeout=20)
            self.path2score.append(result)
            count += 1
            self.progress_count = float(count) / self.total_num
            if self.progress_count == 1:
                break

        # for proc in proc_pool:
        #     proc.join()

        self.path2score.sort(key=lambda x: x[1], reverse=True)
        # save temp score result
        if not os.path.isdir("./temp"):
            os.makedirs("./temp")
        with open("./temp/{}_{}_score.txt".format(self.strategy, self.task_id), "w") as f:
            path2score = [" ".join(list(map(str, x))) + "\n" for x in self.path2score]
            f.writelines(path2score)

        self.selected_img_list = self.select()
        with open(self.selected_img_list_path, 'w') as f:
            f.writelines(self.selected_img_list)
        self.is_done = True
