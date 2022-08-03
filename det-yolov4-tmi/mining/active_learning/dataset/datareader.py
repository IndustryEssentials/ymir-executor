import multiprocessing
from time import sleep
import cv2
import time
from mxnet.gluon.data import dataset
from mxnet import nd, image, gluon
from mxnet.gluon.data.vision import transforms
import mxnet as mx

# from .numpy_mmap_queue import NumpyMmapQueue


class DataReader:
    """
    A class to read the data from a disk asynchronously.
    """
    def __init__(self, img_path_list, max_queue_size=512, num_workers=28):
        self.img_path_list = img_path_list
        self.out_idx = 0
        self.max_queue_size = max_queue_size
        self.queue = multiprocessing.Queue(maxsize=max_queue_size)
        # self.queue = NumpyMmapQueue(
        #     data_name="al_read_img_cache.data", label_name="al_cache.data", max_queue_size, (3, 2048, 2048), (3, 1, 1)
        # )
        self.num_workers = num_workers
        self.idxs = [0 for _ in range(self.num_workers)]
        self.img_path_per_work_list = []
        self.num_imgs_per_worker = int(len(self.img_path_list) / self.num_workers)
        last_i = 0
        for i in range(self.num_workers - 1):
            self.img_path_per_work_list.append(self.img_path_list[i * self.num_imgs_per_worker: (i + 1) * self.num_imgs_per_worker])
            last_i = i
        self.img_path_per_work_list.append(self.img_path_list[(last_i + 1) * self.num_imgs_per_worker:])

    def start(self):
        for i in range(self.num_workers):
            worker = multiprocessing.Process(target = self.run, args = (i, ))
            worker.daemon = True
            worker.start()

    def run(self, worker_id, verbose=False):
        while not self.idxs[worker_id] == len(self.img_path_per_work_list[worker_id]):
            if self.queue.qsize() > self.max_queue_size:
                sleep(0.01)
                continue
            self.read_next(worker_id)
        if verbose:
            print('DataReader {} stopped!'.format(worker_id))

    def read_next(self, worker_id):
        img_path = self.img_path_per_work_list[worker_id][self.idxs[worker_id]].strip()
        # t = time.time()
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError("img read fail {}".format(img_path))
        # print("read img: {}".format(time.time() - t))
        self.idxs[worker_id] += 1
        self.queue.put(obj=(img, img_path), block=True, timeout=None)
        # self.queue.put(img, img_path)

    def dequeue(self, verbose=False):
        while self.queue.qsize() == 0:
            if verbose:
                print('Waiting for data')
        # t = time.time()
        img, img_path = self.queue.get(block=True, timeout=None)
        # print("dequeue: {}".format(time.time() - t))
        self.out_idx += 1
        stop = self.out_idx == len(self.img_path_list)
        return img, img_path, stop


class ImageFolderDataset(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure.

    like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    items : list of tuples
        List of all images in (filename) pairs.
    """
    def __init__(self, img_list, flag=1, transform=None):
        self._flag = flag
        self.img_list = img_list
        self._transform = transform

    def __getitem__(self, idx):
        img = image.imread(self.img_list[idx], self._flag)
        label = idx
        if self._transform is not None:
            return self._transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)



if __name__ == "__main__":
    img_paths = open("./img.txt").readlines()
    img_paths = [each.strip() for each in img_paths]
    transform = transforms.Compose([
        transforms.Resize(608),
        transforms.ToTensor()])
    dataset = gluon.data.DataLoader(ImageFolderDataset(img_paths).transform_first(transform),
                                    batch_size=16, shuffle=False, num_workers=16)
    import time
    total_time = 0
    ctx = [mx.gpu(1), mx.gpu(2), mx.gpu(3)]
    for i, batch in enumerate(dataset):
        start = time.time()
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = batch[1]
        elapsed_time = time.time() - start
        total_time += elapsed_time
        print(i, elapsed_time) 
    print(total_time)
