from mxnet import nd
import os


class ALDD:
    """
    implement ALDD, Active Learning for Deep Detection Neural Networks(ICCV2019)
    """
    def __init__(
        self, avg_pool_size=3, max_pool_size=12, model=None, labeled_dataset=None
    ):
        self.avg_pool_size = avg_pool_size
        self.max_pool_size = max_pool_size
        assert hasattr(model, "get_heatmap")
        self.model = model


    def compute_score(self, imgs, return_vis=False):
        """
        args:
            imgs: list[np.array(H, W, C)]
        returns:
            scores: list of float
        """
        if return_vis:
            heatmap, c_batch, s_batch = self.model.get_heatmap(imgs, return_vis)
        else:
            heatmap = self.model.get_heatmap(imgs)

        kernel = self.avg_pool_size
        pad = (kernel - 1) // 2

        # mean of entropy
        ent = -heatmap * nd.log(heatmap + 1e-12)  # N, C, H, W
        ent = nd.sum(ent, axis=1, keepdims=True)  # N, 1, H, W
        mean_of_entropy = nd.Pooling(ent, kernel = (kernel, kernel), pool_type = 'avg', stride = (1, 1), pad = (pad, pad))  # N, 1, H, W

        # entropy of mean
        mean_heatmap = nd.Pooling(heatmap, kernel = (kernel, kernel), pool_type = 'avg', stride = (1, 1), pad = (pad, pad))  # N, C, H, W
        entropy_of_mean = -mean_heatmap * nd.log(mean_heatmap + 1e-12)  # N, C, H, W
        entropy_of_mean = nd.sum(entropy_of_mean, axis=1, keepdims=True)  # N, 1, H, W
        uncertainty = entropy_of_mean - mean_of_entropy

        # aggregating
        kernel = self.max_pool_size
        pad = 0
        pooled_uncertainty = nd.Pooling(uncertainty, kernel = (kernel, kernel), pool_type = 'max', stride = (kernel, kernel), pad = (pad, pad))  # N, 1, H, W
        scores = nd.mean(pooled_uncertainty, axis=[1, 2, 3])
        scores = scores.asnumpy()
        if return_vis:
            return heatmap, mean_of_entropy, entropy_of_mean, uncertainty, scores, c_batch, s_batch
        else:
            return scores
