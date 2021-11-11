import os
import cv2 
import pdb

if __name__ == "__main__":
    files = open("/out/img.txt", 'r').readlines()
    files = [each.strip() for each in files]

    for i, each_imgpath in enumerate(files):
        if i % 1000 == 0:
            print("converted {} image annotations".format(i))
        each_txtfile = os.path.splitext(each_imgpath)[0] + ".txt" 
        if not os.path.isfile(each_txtfile):
            txt_out_f = open(each_txtfile, 'w')
            txt_out_f.close()
            continue
        img = cv2.imread(each_imgpath)
        if img is None:
            raise ValueError("can not read image: {}".format(each_imgpath))
        img_h, img_w, _  = img.shape
        txt_content = open(each_txtfile, 'r').readlines()
        txt_content = [each.strip() for each in txt_content]
        output_list = []
        if len(txt_content) < 1:
            txt_out_f = open(each_txtfile, 'w')
            txt_out_f.close()
            continue
        for each_line in txt_content:
            each_line = [int(each) for each in each_line.split(",")]
            cls, xmin, ymin, xmax, ymax = each_line
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            if bbox_w < 10 or bbox_h < 10:
                print("bbox too small {} {} \n {}".format(bbox_w, bbox_h, each_imgpath))
                continue
            xcenter /= img_w
            ycenter /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            output_str = "{} {} {} {} {}\n".format(cls, xcenter, ycenter, bbox_w, bbox_h)
            output_list.append(output_str)
        txt_out_f = open(each_txtfile, 'w')
        for each_str in output_list:
            txt_out_f.write(each_str)
        txt_out_f.close()

