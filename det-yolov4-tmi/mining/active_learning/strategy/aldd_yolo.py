from mxnet import nd
import os
import numpy as np

class ALDD_YOLO:
    """
    implement ALDD, Active Learning for Deep Detection Neural Networks(ICCV2019)
    """
    def __init__(self, avg_pool_size=9, max_pool_size=30, model=None):
        self.avg_pool_size = avg_pool_size
        self.max_pool_size = max_pool_size
        assert hasattr(model, "get_heatmap")
        self.model = model

    def calc_unc_val(self, heatmap):
        kernel = self.avg_pool_size
        pad = (kernel - 1) // 2
        
        # mean of entropy
        prob_pixel = heatmap
        prob_pixel_m1 = 1 - heatmap
        ent = -(prob_pixel * nd.log(prob_pixel + 1e-12) + prob_pixel_m1 * nd.log(prob_pixel_m1 + 1e-12)) # N, C, H, W
        ent = nd.sum(ent, axis=1, keepdims=True)  # N, 1, H, W
        mean_of_entropy = nd.Pooling(ent, kernel = (kernel, kernel), pool_type = 'avg', stride = (1, 1), count_include_pad=False, pad = (pad, pad))  # N, 1, H, W

        # entropy of mean
        prob_local = nd.Pooling(heatmap, kernel = (kernel, kernel), pool_type = 'avg', stride = (1, 1), count_include_pad=False, pad = (pad, pad))  # N, C, H, W
        prob_local_m1 = 1 - prob_local
        entropy_of_mean = -(prob_local * nd.log(prob_local + 1e-12) + prob_local_m1 * nd.log(prob_local_m1 + 1e-12))  # N, C, H, W
        entropy_of_mean = nd.sum(entropy_of_mean, axis=1, keepdims=True)  # N, 1, H, W

        uncertainty = entropy_of_mean - mean_of_entropy
        unc = nd.Pooling(uncertainty, kernel = (self.max_pool_size, self.max_pool_size), pool_type = 'max', stride = (30, 30), count_include_pad=False, pad = (2, 2))

        # aggregating
        scores = nd.mean(unc, axis=(1, 2, 3))
        return scores

    def compute_score(self, imgs, return_vis=False):
        """
        args:
            imgs: list[np.array(H, W, C)]
        returns:
            scores: list of float
        """
        net_output = self.model.get_heatmap(imgs)
        net_output_conf = net_output[:, :, 4]
        total_scores = []

        for each_class_index in range(self.model.num_of_class):
            net_output_cls_mult_conf = net_output_conf * net_output[:, :, 5 + each_class_index]
            feature_map_1 = net_output_cls_mult_conf[:, :self.model.strides[0]]
            feature_map_1 = nd.reshape(feature_map_1, [-1, self.model.featuremap_outshape[0], self.model.featuremap_outshape[0], 3])
            feature_map_1 = nd.transpose(feature_map_1, [0, 3, 1, 2])
            feature_map_1 = nd.contrib.BilinearResize2D(feature_map_1, height=self.model.input_dim, width=self.model.input_dim)

            feature_map_2 = net_output_cls_mult_conf[:, self.model.strides[0]:self.model.strides[1]]
            feature_map_2 = nd.reshape(feature_map_2, [-1, self.model.featuremap_outshape[1], self.model.featuremap_outshape[1], 3])
            feature_map_2 = nd.transpose(feature_map_2, [0, 3, 1, 2])
            feature_map_2 = nd.contrib.BilinearResize2D(feature_map_2, height=self.model.input_dim, width=self.model.input_dim)

            feature_map_3 = net_output_cls_mult_conf[:, self.model.strides[1]:self.model.strides[2]]
            feature_map_3 = nd.reshape(feature_map_3, [-1, self.model.featuremap_outshape[2], self.model.featuremap_outshape[2], 3])
            feature_map_3 = nd.transpose(feature_map_3, [0, 3, 1, 2])
            feature_map_3 = nd.contrib.BilinearResize2D(feature_map_3, height=self.model.input_dim, width=self.model.input_dim)

            heatmap = nd.concatenate([feature_map_1, feature_map_2, feature_map_3], 1)
            scores = self.calc_unc_val(heatmap)
            scores = scores.asnumpy()
            total_scores.append(scores)
        
        total_scores = np.array(total_scores)
        total_scores = total_scores * self.model.class_distribution_score
        total_scores = np.sum(total_scores, axis=0)

        return total_scores
