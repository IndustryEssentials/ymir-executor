# use single run first 1000 iter and stop
./darknet detector train ./cfg/coco.data ./cfg/yolov4_iter1000.cfg ./yolov4.conv.137 -map -gpus 0 -dont_show
# after first 1000 iter then use 4 gpus to train model
./darknet detector train ./cfg/coco.data ./cfg/yolov4.cfg /out/models/yolov4_iter1000_last.weights -map -gpus 0,1,2,3 -dont_show
