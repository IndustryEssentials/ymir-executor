import mxnet as mx
from mxnet import nd, gluon
import numpy as np


class YoloNet:
    def __init__(self, weights_file, input_dim=608, ctx=mx.cpu(), class_distribution_score=np.array([1.0]), num_of_class=1):
        self.ctx = ctx
        self.input_dim = input_dim
        self.class_distribution_score = class_distribution_score
        self.num_of_class = num_of_class
        symbol_file = weights_file.replace(weights_file.split('-')[-1], 'symbol.json')
        self.net = gluon.SymbolBlock.imports(symbol_file, ['data'], weights_file, ctx = self.ctx)
        self.net.hybridize()
        self.featuremap_outshape = [input_dim // 32, input_dim // 32 * 2, input_dim // 32 * 2 * 2]
        self.strides = [self.featuremap_outshape[0]**2*3,
                        self.featuremap_outshape[0]**2*3 + self.featuremap_outshape[1]**2*3,
                        self.featuremap_outshape[0]**2*3 + self.featuremap_outshape[1]**2*3 + self.featuremap_outshape[2]**2*3]

    def get_heatmap(self, img_arr):
        results = self.net(img_arr)

        return results
