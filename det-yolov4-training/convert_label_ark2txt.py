import os

import cv2


def _annotation_path_for_image(image_path: str, annotations_dir: str) -> str:
    # replace dir
    annotation_path = image_path.replace('/in/cache', annotations_dir, 1)
    annotation_path = annotation_path.replace('/in/train', annotations_dir, 1)
    annotation_path = annotation_path.replace('/in/val', annotations_dir, 1)
    # replace ext
    annotation_path = os.path.splitext(annotation_path)[0] + '.txt'
    return annotation_path


def _convert_annotations(index_file_path: str, src_annotations_dir: str, dst_annotations_dir: str) -> None:
    """
    read all images and annotations in `index_file_path`, change annotations format, write to `annotations_dir`

    `annotations_dir` should exists
    """
    with open(index_file_path, 'r') as f:
        files = f.readlines()
        files = [each.strip() for each in files]

    for i, each_imgpath in enumerate(files):
        if i % 1000 == 0:
            print(f"converted {i} image annotations")

        each_txtfile = _annotation_path_for_image(image_path=each_imgpath, annotations_dir=src_annotations_dir)

        # if no annotations for that image, create an empty one
        if not os.path.isfile(each_txtfile):
            open(each_txtfile, 'w').close()
            continue

        img = cv2.imread(each_imgpath)
        if img is None:
            raise ValueError(f"can not read image: {each_imgpath}")
        img_h, img_w, _ = img.shape

        with open(each_txtfile, 'r') as f:
            txt_content = f.readlines()
            txt_content = [each.strip() for each in txt_content]
        output_list = []

        for each_line in txt_content:
            each_line = [int(each) for each in each_line.split(",")]

            cls, xmin, ymin, xmax, ymax, *_ = each_line
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            if bbox_w < 10 or bbox_h < 10:
                # too small, ignored
                continue

            xcenter /= img_w
            ycenter /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            output_str = f"{cls} {xcenter} {ycenter} {bbox_w} {bbox_h}"
            output_list.append(output_str)

        # write output_list
        dst_txtfile = _annotation_path_for_image(image_path=each_imgpath, annotations_dir=dst_annotations_dir)
        os.makedirs(os.path.dirname(dst_txtfile), exist_ok=True)
        with open(dst_txtfile, 'w') as f:
            for each_str in output_list:
                f.write(f"{each_str}\n")


if __name__ == "__main__":
    _convert_annotations(index_file_path='/in/train/index.tsv', src_annotations_dir='/in/train',
                         dst_annotations_dir='/in/tmp_labels')
    _convert_annotations(index_file_path='/in/val/index.tsv', src_annotations_dir='/in/val',
                         dst_annotations_dir='/in/tmp_labels')
