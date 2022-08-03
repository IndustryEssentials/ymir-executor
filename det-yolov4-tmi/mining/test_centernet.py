import psutil
import time
from tqdm import tqdm
from active_learning.dataset import DataReader
from active_learning.model_inference import CenterNet


proc = psutil.Process()
proc.cpu_affinity([0])
weight_file = './mobilenet_sc_cpu_combined_V1.4-0130.params'
# weight_file = '/data/zengzhuoxi/UbiquitousDetector/centernet-mx/deploy_model/mobilenet_sc_cpu_combine_baseline_v0-0130.params'
classes_file = './combined_class.txt'
gpu_id = '0'
confidence_thresh = 0.1
batch_size = 1
nms_thresh = 0.45
input_dim = 512
output_dim = 128
net = CenterNet(
    weight_file,
    classes_file,
    gpu_id,
    confidence_thresh,
    batch_size,
    nms_thresh,
    input_dim,
    output_dim,
    mode = 'combined'
)
# img_path = "/data/zengzhuoxi/UbiquitousDetector/MultiDetector_38/own_test/own_TestSet/ordinary_hour_day_without_rain_zebra_crossing_20190415_162616_6784.jpg"
# img = cv2.imread(img_path)
# result, num_bbox = net.detect([img])
# print(result, num_bbox)
# print(len(result), len(result[0]))
# net.draw_bbox(result, num_bbox, [img_path], "result/", mode = "combined")
# print 'done'
with open("/home/xiongzihua/code/active_learning/train_data_path/unlabeled_base.txt", 'r') as f:
    lines = f.readlines()

data_reader = DataReader(lines[19000:20000])
data_reader.start()
imgs = []
total = 100
bin_count = 1000 / 100
count = 0
t = time.time()
with tqdm(total=total, desc='Progress', ncols=total, ascii=' =') as pbar:
    while True:
        img, img_path, stop = data_reader.dequeue()
        # hm = net.get_heatmap_v2([img])
        preds, num_bbox = net.detect([img])
        count += 1
        if count % bin_count == 0:
            pbar.update(1)
        net.save_result(preds, dst_dir="./centernet_result", img_path_list=[img_path])
        if stop:
            break
print((time.time() - t))

# temp = lines[18000:19000]
# t = time.time()
# for i, img_path in enumerate(temp):
#     img_path = img_path.strip()
#     hm = net.get_heatmap([img_path])
# print((time.time() - t))



# print(hm)


# weight_file = '/home/huangdewei/detector/models/headshoulder/mobilenet_sc_nnie_headshoulder_V0.4_sampled-0130.params'
# classes_file = '/home/huangdewei/detector/headshoulder_class.txt'
# gpu_id = '0'
# confidence_thresh = 0.1
# batch_size = 1
# nms_thresh = 0.45
# input_dim = 512
# output_dim = 128
# # net = darknet_mxnet(weight_file, classes_file, gpu_id, confidence_thresh, batch_size, nms_thresh, input_dim)
# net = CenterNet(weight_file,
#                       classes_file,
#                       gpu_id,
#                       confidence_thresh,
#                       batch_size,
#                       nms_thresh,
#                       input_dim,
#                       output_dim,
#                       mode = 'normal')
# img = cv2.imread(
#     '/data1/zengzhuoxi/data/bus_project_data_center/0417/0417_TestSet/001574_ExternDisk_ch6_20200417180004_20200417190004.jpg')
# # result = net.detect([np.random.uniform(0, 255, [1080, 1920, 3]), np.random.uniform(0, 255, [1080, 1920, 3]))
# result, num_bbox = net.detect([img])
# print(result, num_bbox)
# print(len(result), len(result[0]))
# img_path = '/data1/zengzhuoxi/data/bus_project_data_center/0417/0417_TestSet/001574_ExternDisk_ch6_20200417180004_20200417190004.jpg'
# net.draw_bbox(result, num_bbox, [img_path], "result/", mode = "normal")
# print 'done'