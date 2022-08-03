./darknet detector train ./cfg/coco.data ./cfg/yolov3.cfg ./models/yolov3_last.weights -map -gpus 4,5,6,7 -dont_show
