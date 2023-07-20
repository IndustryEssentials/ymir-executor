import mxnet as mx
from mxnet import nd, gluon
import numpy as np
import cv2
import os
import time


class CenterNet:
    def __init__(
            self, weights_file, classes_file='./combined_class.txt', gpu_id = '0', confidence_thresh = 0.2, batch_size = 1,
            nms_thresh = 0.45, input_dim = 512, output_dim = 128, mode = 'combined'
    ):
        self.classes = list(map(lambda x: x.strip(), open(classes_file, 'r').read().split('\n')))
        
        # TODO: add swtich between cpu and gpu.
        self.ctx = mx.cpu() # [mx.gpu(i) for i in self.gpu][0]
        
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conf_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.mode = mode
        symbol_file = weights_file.replace(weights_file.split('-')[-1], 'symbol.json')
        self.net = gluon.SymbolBlock.imports(symbol_file, ['data'], weights_file, ctx = self.ctx)
        self.net.hybridize()

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype = np.float32)

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def transform_preds(self, coords, center, scale, output_size):
        trans = self.get_affine_transform(center, scale, 0, output_size, inv = 1)
        target_coords = self.affine_transform(coords, trans)
        return target_coords

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.], dtype = np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def get_affine_transform(self,
                             center,
                             scale,
                             rot,
                             output_size,
                             shift = np.array([0, 0], dtype = np.float32),
                             inv = 0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype = np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype = np.float32)
        dst = np.zeros((3, 2), dtype = np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def prep_image(self, img, input_dim):
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype = np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = input_dim, input_dim

        # affine shift and scales
        trans_input = self.get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags = cv2.INTER_LINEAR)
        # normalize to -1 ~ 1
        inp = (inp.astype(np.float32) - 127.5) / 127.5
        # inp = (inp - mean) / std

        # (h,w,c)-->(c,h,w)
        inp = inp.transpose(2, 0, 1)

        s = np.array([s])
        wh = np.array([height, width])

        return inp, c, s, wh

    def find_keypoints(self, hm, kernel = 3):
        pad = (kernel - 1) // 2
        hm_max = mx.nd.Pooling(hm, kernel = (kernel, kernel), pool_type = 'max', stride = (1, 1), pad = (pad, pad))
        mask = (hm == hm_max)

        return hm * mask

    def find_topk(self, hm, batch_size, num_classes, output_dims, K = 40):
        # (b,c,h,w)-->(b,c*h*w)
        hm1 = mx.nd.reshape(hm, shape = (batch_size, -1))
        # (b,c*h*w)-->(b,k)
        topk_scores, topk_inds = mx.nd.topk(hm1, ret_typ = 'both', k = K)
        # cls--(b,k)
        topk_cls = topk_inds / (output_dims * output_dims)
        # topk_cls = mx.nd.cast(topk_cls, dtype='uint8')

        # xs, ys--(b,k)
        topk_inds = topk_inds % (output_dims * output_dims)
        topk_ys = (topk_inds / output_dims)
        topk_xs = (topk_inds % output_dims)
        topk_ys = mx.nd.floor(topk_ys)
        topk_xs = mx.nd.floor(topk_xs)

        return topk_scores, topk_inds, topk_cls, topk_xs, topk_ys

    def find_topk_combined(self, hm, quality, batch_size, num_classes, output_dims, K = 40):
        # (b,c,h,w)-->(b,c*h*w)
        hm1 = mx.nd.reshape(hm, shape = (batch_size, -1))
        quality1 = mx.nd.reshape(quality, shape = (batch_size, -1))
        # (b,c*h*w)-->(b,k)
        topk_scores, topk_inds = mx.nd.topk(hm1, ret_typ = 'both', k = K)
        topk_quality = mx.nd.zeros(topk_inds.shape).as_in_context(topk_inds.context)
        for b in range(batch_size):
            topk_quality[b, :] = quality1[b, topk_inds[b]]
        # cls--(b,k)
        topk_cls = topk_inds / (output_dims * output_dims)
        # topk_cls = mx.nd.cast(topk_cls, dtype='uint8')

        # xs, ys--(b,k)
        topk_inds = topk_inds % (output_dims * output_dims)
        topk_ys = (topk_inds / output_dims)
        topk_xs = (topk_inds % output_dims)
        topk_ys = mx.nd.floor(topk_ys)
        topk_xs = mx.nd.floor(topk_xs)
        
        cls_scores = mx.nd.zeros((batch_size, K, num_classes)).as_in_context(topk_inds.context)
        for b in range(batch_size):
            cls_scores[b, :, :] = hm[b, :, topk_ys[b], topk_xs[b]]

        return topk_scores, topk_inds, topk_quality, topk_cls, topk_xs, topk_ys, cls_scores

    def gather_wh_offset(self, wh, topk_inds, batch_size, K = 40):
        zeros_array = mx.nd.zeros(shape = (1, K), ctx = wh.context)  # row0 index
        ones_array = mx.nd.ones(shape = (1, K), ctx = wh.context)  # row1 index

        topk_w = None
        topk_h = None
        # (b,2,128,128)--->(b,2,128*128)
        wh1 = mx.nd.reshape(wh, shape = (batch_size, 2, -1))
        for b in range(batch_size):
            # (b,2,128*128)-->(1, 2, 128*128)
            # wh_b = mx.sym.slice_axis(wh1, axis=0, begin=b, end=b+1)
            wh_b = wh1[b, :, :]
            # (1,2,128*128)-->(2,128*128)
            wh_b = mx.nd.reshape(wh_b, shape = (2, -1))

            # (b,k)-->(1, k)
            # topk_inds_b = mx.sym.slice_axis(topk_inds, axis=0, begin=b, end=b+1)
            topk_inds_b = topk_inds[b, :]
            topk_inds_b = mx.nd.expand_dims(topk_inds_b, axis = 0)

            # (2,k)
            topk_inds_b_row0 = mx.nd.concat(zeros_array, topk_inds_b, dim = 0)
            # (2,128,128)-->(k,)
            topk_w_b = mx.nd.gather_nd(wh_b, topk_inds_b_row0)

            # (2,k)
            topk_inds_b_row1 = mx.nd.concat(ones_array, topk_inds_b, dim = 0)
            # (2,128,128)-->(k,)
            topk_h_b = mx.nd.gather_nd(wh_b, topk_inds_b_row1)

            # (1,k)
            topk_w_b = mx.nd.expand_dims(topk_w_b, axis = 0)
            topk_h_b = mx.nd.expand_dims(topk_h_b, axis = 0)
            # (1,k)-->(b,k)
            if b == 0:
                topk_w = topk_w_b
                topk_h = topk_h_b
            else:
                topk_w = mx.nd.concat(topk_w, topk_w_b, dim = 0)
                topk_h = mx.nd.concat(topk_h, topk_h_b, dim = 0)

        return topk_w, topk_h

    def CenterNet_get_bbox(self, hm, wh, offset, batch_size, num_classes, output_dims, kernel = 3, K = 40):
        hm_max = self.find_keypoints(hm, kernel)
        # print(hm_max)
        # (b,k)
        topk_scores, topk_inds, topk_cls, topk_xs, topk_ys = self.find_topk(hm_max, batch_size, num_classes,
                                                                            output_dims, K = K)
        # (b,k), (b,k)
        topk_w, topk_h = self.gather_wh_offset(wh, topk_inds, batch_size, K = K)
        topk_dx, topk_dy = self.gather_wh_offset(offset, topk_inds, batch_size, K = K)
        # (b,k)
        topk_xs = topk_xs + topk_dx
        topk_ys = topk_ys + topk_dy
        # (b,k)
        topk_w_05 = topk_w / 2
        topk_h_05 = topk_h / 2
        # (x,y,w,h)-->(xmin, ymin, xmax, ymax)
        xmin = topk_xs - topk_w_05
        xmax = topk_xs + topk_w_05
        ymin = topk_ys - topk_h_05
        ymax = topk_ys + topk_h_05

        # (b,k)-->(b,k,1)
        xmin = mx.nd.expand_dims(xmin, axis = 2)
        xmax = mx.nd.expand_dims(xmax, axis = 2)
        ymin = mx.nd.expand_dims(ymin, axis = 2)
        ymax = mx.nd.expand_dims(ymax, axis = 2)
        topk_scores = mx.nd.expand_dims(topk_scores, axis = 2)
        topk_cls = mx.nd.expand_dims(topk_cls, axis = 2)

        # (b,k,1)-->(b,k,3)
        bbox = mx.nd.concat(topk_cls, topk_scores, xmin, ymin, xmax, ymax, dim = 2)

        return bbox

    def CenterNet_get_bbox_combined(self, hm, wh, offset, quality, batch_size, num_classes, output_dims, K = 40):
        cls_max_score = hm.max(axis=(2, 3))
        hm_max = self.find_keypoints(hm)
        # print(hm_max)
        # (b,k)
        topk_scores, topk_inds, topk_quality, topk_cls, topk_xs, topk_ys, cls_scores = self.find_topk_combined(hm_max, quality,
                                                                                                   batch_size,
                                                                                                   num_classes,
                                                                                                   output_dims,
                                                                                                   K = K)

        topk_quality = mx.nd.clip(topk_quality, 0, 1)

        # (b,k), (b,k)
        if self.mode == 'combined':
            topk_w, topk_h = self.gather_wh_offset(wh, topk_inds, batch_size, K = K)
            topk_dx, topk_dy = self.gather_wh_offset(offset, topk_inds, batch_size, K = K)
        elif self.mode == 'combined_ind':
            topk_dx, topk_dy = self.gather_wh_offset(offset, topk_inds, batch_size, K = K)
            wh = wh.reshape([batch_size, num_classes * 2, -1])
            topk_w = mx.nd.zeros([batch_size, K]).as_in_context(topk_dx.context)
            topk_h = mx.nd.zeros([batch_size, K]).as_in_context(topk_dx.context)
            for b in range(batch_size):
                topk_w[b, :] = wh[b, topk_cls[b].astype(np.int) * 2, topk_inds[b]]
                topk_h[b, :] = wh[b, topk_cls[b].astype(np.int) * 2 + 1, topk_inds[b]]
        # (b,k)
        topk_xs = topk_xs + topk_dx
        topk_ys = topk_ys + topk_dy
        # (b,k)
        topk_w_05 = topk_w / 2
        topk_h_05 = topk_h / 2
        # (x,y,w,h)-->(xmin, ymin, xmax, ymax)
        xmin = topk_xs - topk_w_05
        xmax = topk_xs + topk_w_05
        ymin = topk_ys - topk_h_05
        ymax = topk_ys + topk_h_05
        # (b,k)-->(b,k,1)
        xmin = mx.nd.expand_dims(xmin, axis = 2)
        xmax = mx.nd.expand_dims(xmax, axis = 2)
        ymin = mx.nd.expand_dims(ymin, axis = 2)
        ymax = mx.nd.expand_dims(ymax, axis = 2)
        topk_scores = mx.nd.expand_dims(topk_scores, axis = 2)
        topk_cls = mx.nd.expand_dims(topk_cls, axis = 2)
        topk_quality = mx.nd.expand_dims(topk_quality, axis = 2)

        # (b,k,1)-->(b,k,3)
        # print(e)
        bbox = mx.nd.concat(topk_cls, topk_scores, xmin, ymin, xmax, ymax, topk_quality, cls_scores, dim = 2)
        return bbox, cls_max_score

    def iou_calc(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)
        box2Area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)

        iou = inter / float(box1Area + box2Area - inter + 1e-14)

        return iou

    def nms(self, boxes):
        i = 0
        while (True):
            if i >= len(boxes):
                break
            if boxes[i][1] == 0:
                boxes = np.delete(boxes, i, 0)
            else:
                i += 1

        if len(boxes) == 0:
            return []
        elif boxes.shape[0] == 1:
            return boxes
        boxes = sorted(boxes, key = lambda x: -x[1])
        # for i in range(len(boxes)):
        #     result
        #     for j in range(i + 1, len(boxes)):
        #         iou = iou_calc(boxes[i], boxes[j])
        #         if iou <= nms_thresh:
        #             boxes.pop()
        i = 0
        j = i + 1
        while (True):
            # j += 1
            iou = self.iou_calc(boxes[i][2:], boxes[j][2:])
            if iou >= self.nms_thresh:
                boxes.pop(j)
                if j >= (len(boxes)):
                    i += 1
                    j = i
            else:
                j += 1

                if j >= (len(boxes)):
                    i += 1
                    j = i + 1
            # print '%i, %i'%(i, j)
            # print '-'*10
            if i >= (len(boxes) - 1):
                break

        return boxes

    def post_process(self, preds, batch_size, c_batch, s_batch, wh_batch, conf_thresh = 0.5):
        # (b,k)
        conf_mask = (preds[:, :, 1] > conf_thresh)
        # (b,k)-->(b,)
        num_bbox = mx.nd.sum(conf_mask, axis = 1)
        # (b,k)-->(b,k,1)
        conf_mask = conf_mask.expand_dims(2)
        # (b,k,6)*(b,k,1)=(b,k,6)
        preds = preds * conf_mask

        preds = preds.asnumpy()
        num_bbox = num_bbox.asnumpy()
        for b in range(batch_size):
            for n in range(int(num_bbox[b])):
                xy = preds[b, n, 2:4]
                xy = self.transform_preds(xy, c_batch[b], s_batch[b], [128, 128])
                preds[b, n, 2:4] = list(map(int, xy))

                xy = preds[b, n, 4:6]
                xy = self.transform_preds(xy, c_batch[b], s_batch[b], [128, 128])
                preds[b, n, 4:6] = list(map(int, xy))

                preds[b, n, 0] = int(preds[b, n, 0])

        bbox = []
        num_bbox = []
        for b in range(batch_size):
            bbox.append(self.nms(preds[b]))
            # bbox.append(self.nms(preds[b, :3]))
            num_bbox.append(len(bbox[-1]))
        return bbox, num_bbox

    def detect(self, imlist):
        if not imlist:
            print("no images to detect")
            exit()
        leftover = 0
        if len(imlist) % self.batch_size:
            leftover = 1
        num_batches = len(imlist) // self.batch_size + leftover
        # print('num_batches:', num_batches)
        im_batches = [imlist[i * self.batch_size: min((i + 1) * self.batch_size, len(imlist))] for i in
                      range(num_batches)]

        t = 0
        boxes = None
        for batch in im_batches:
            img_list = [img for img in batch]

            tmp_batch = list(map(self.prep_image, img_list, [self.input_dim for x in range(len(batch))]))
            img_batch = [x[0] for x in tmp_batch]
            c_batch = [x[1] for x in tmp_batch]
            s_batch = [x[2] for x in tmp_batch]
            wh_batch = [x[3] for x in tmp_batch]
            # print(len(img_batch))
            img_batch = mx.nd.array(img_batch, ctx = self.ctx)
            # img_batch = mx.nd.array(img_batch)
            # img_batch = gluon.utils.split_and_load(img_batch, ctx_list=self.ctx, batch_axis=0)

            results = self.net(img_batch)

            hm_out = results[0]
            wh_out = results[1]
            offset_out = results[2]
            if self.mode == 'combined' or self.mode == 'combined_ind':
                quality_out = results[3]
                preds, cls_max_score = self.CenterNet_get_bbox_combined(hm_out, wh_out, offset_out, quality_out, len(batch),
                                                         len(self.classes), self.output_dim, K = 200)
                preds, num_bbox = self.post_process(preds, len(batch), c_batch, s_batch, wh_batch,
                                                    conf_thresh = self.conf_thresh)
            else:
                preds = self.CenterNet_get_bbox(hm_out, wh_out, offset_out, len(batch),
                                                len(self.classes), self.output_dim, 3, K = 200)
                preds, num_bbox = self.post_process(preds, len(batch), c_batch, s_batch, wh_batch,
                                                    conf_thresh = self.conf_thresh)

        return preds, cls_max_score, num_bbox

    # def get_heatmap_v2(self, img_path_list):
    #     img_list = []
    #     for img_path in img_path_list:
    #         img = cv2.imread(img_path)
    #         img_list.append(img)

    #     tmp_batch = list(map(self.prep_image, img_list, [self.input_dim] * len(img_list)))
    #     img_batch = [x[0] for x in tmp_batch]
    #     c_batch = [x[1] for x in tmp_batch]
    #     s_batch = [x[2] for x in tmp_batch]
    #     wh_batch = [x[3] for x in tmp_batch]
    #     img_batch = mx.nd.array(img_batch, ctx = self.ctx)

    #     results = self.net(img_batch)

    #     hm_out = results[0]
    #     return hm_out

    def get_cald_input(self, img_list):
        preds, cls_max_score, num_bbox = self.detect(img_list)
        # preds = [pred.expand_dims(axis=0) for pred in preds]
        # preds = mx.nd.concat(*preds, dim=0)
        preds = preds[0]
        if len(preds) == 0:
            return None, None, None, None, None, None, False
        preds = [np.expand_dims(pred, axis=0) for pred in preds]
        preds = np.concatenate(preds)
        cls, max_scores, boxes, quality, cls_scores = preds[:, :1], preds[:, 1:2], preds[:, 2:6],\
            preds[:, 6:7], preds[:, 7:]
        return cls, max_scores, boxes, quality, cls_scores, cls_max_score, True

    def get_heatmap(self, img_list, return_vis=False):
        tmp_batch = list(map(self.prep_image, img_list, [self.input_dim] * len(img_list)))
        img_batch = [x[0] for x in tmp_batch]
        c_batch = [x[1] for x in tmp_batch]
        s_batch = [x[2] for x in tmp_batch]
        wh_batch = [x[3] for x in tmp_batch]
        img_batch = mx.nd.array(img_batch, ctx = self.ctx)
        print("before forward")
        results = self.net(img_batch)
        print("return forward result")

        # hack fix.
        if len(results) > 4: # Resnet18
            hm_out = results[3]
        else: # mobilenet
            hm_out = results[0]
        if return_vis:
            return hm_out, c_batch, s_batch
        else:
            return hm_out

    def draw_bbox(self, preds, num_bbox, batch, dst_dir):
        if dst_dir[-1] != '/':
            dst_dir += '/'

        for b in range(len(batch)):
            img_path = batch[b]
            img = cv2.imread(img_path)
            bbox = preds[b]  # (k,6)
            # bbox = nms(bbox, self._nms_thresh)
            num_boxes = num_bbox[b]
            for n in range(int(num_boxes)):
                box = bbox[n]
                cls = int(box[0])
                ness = box[1]
                xmin = int(box[2]) if int(box[3]) >= 0 else 0
                ymin = int(box[3]) if int(box[4]) >= 0 else 0
                xmax = int(box[4])
                ymax = int(box[5])
                c1 = tuple([xmin, ymin])
                c2 = tuple([xmax, ymax])
                color = (0, 0, 255)
                label = "{0} {1:.3f}".format(self.classes[cls], ness)
                cv2.rectangle(img, c1, c2, color, 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
                cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 7), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)

            if img_path is not None:
                cv2.imwrite(dst_dir + 'image/' + os.path.basename(img_path), img)
        return img

    def save_result(self, preds, dst_dir, img_path_list):
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for b in range(len(img_path_list)):
            img_path = img_path_list[b]
            bbox = preds[b]  # (k,6)
            img_name = os.path.basename(img_path).strip()
            result_path = os.path.join(dst_dir, img_name + '.txt')
            print(result_path)
            with open(result_path, 'w') as f:
                for n in range(len(bbox)):
                    box = bbox[n]
                    cls = int(box[0])
                    ness = box[1]
                    xmin = int(box[2]) if int(box[3]) >= 0 else 0
                    ymin = int(box[3]) if int(box[4]) >= 0 else 0
                    xmax = int(box[4])
                    ymax = int(box[5])
                    line = '{} {} {} {} {} {}\n'.format(self.classes[cls], ness, xmin, ymin, xmax, ymax)
                    f.write(line)


if __name__ == '__main__':
    import psutil
    from active_learning.dataset import DataReader
    # proc = psutil.Process()
    # proc.cpu_affinity([0])
    # weight_file = '/data/zengzhuoxi/UbiquitousDetector/centernet-mx/deploy_model/mobilenet_sc_cpu_combined_V1.4-0130.params'
    weight_file = '../centernet-mx/deploy_model/mobilenet_sc_cpu_combined_aldd_select_iter2_2000-0130.params'
    # weight_file = '/data/zengzhuoxi/UbiquitousDetector/centernet-mx/deploy_model/mobilenet_sc_cpu_combine_baseline_v0-0130.params'
    classes_file = './combined_class.txt'
    gpu_id = '0'
    confidence_thresh = 0.3
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
    with open("train_data_path/aldd_select_labeled_iter1_2000.txt", 'r') as f:
        lines = f.readlines()

    data_reader = DataReader(lines[:2])
    data_reader.start()
    imgs = []
    t = time.time()
    while True:
        img, img_path, stop = data_reader.dequeue()
        print(img_path)
        # hm = net.get_heatmap_v2([img])
        result, num_bbox = net.get_cald_input([img])
        net.draw_bbox(result, num_bbox, [img_path], "result/")
        if stop:
            break
    print((time.time() - t))

    # temp = lines[2400:2600]
    # t = time.time()
    # for i, img_path in enumerate(temp):
    #     img_path = img_path.strip()
    #     hm = net.get_heatmap([img_path])
    #     if i == 199:
    #         break
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
