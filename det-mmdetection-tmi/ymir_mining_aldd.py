import sys

from easydict import EasyDict as edict
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models.detectors import SingleStageDetector, TwoStageDetector
from ymir_exc.util import get_merged_config

from .mining_base import ALDDMining
from .ymir_infer import YmirModel
from .ymir_mining_random import RandomMiner


class ALDDMiner(RandomMiner):

    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.ymir_model = YmirModel(cfg)
        mmdet_cfg = self.ymir_model.cfg
        mmdet_cfg.data.test.pipeline = replace_ImageToTensor(mmdet_cfg.data.test.pipeline)
        self.test_pipeline = Compose(cfg.data.test.pipeline)
        self.aldd_miner = ALDDMining(cfg, [640, 640])

    def compute_score(self, asset_path: str) -> int:
        dict_data = dict(img_info=dict(filename=asset_path), img_prefix=None)
        pipeline_data = self.test_pipeline(dict_data)
        data = collate([pipeline_data], samples_per_gpu=1)
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        # scatter to specified GPU
        data = scatter(data, [self.device])[0]

        if isinstance(self.ymir_model.model, SingleStageDetector):
            cls_score, bbox_pred = self.ymir_model.model.forward_dummy(data['img'])
            mining_score = self.aldd_miner(bbox_pred)

            return mining_score
        elif isinstance(self.ymir_model.model, TwoStageDetector):
            # (rpn_outs, roi_outs)
            # outs = self.ymir_model.model.forward_dummy(img)
            raise NotImplementedError('aldd mining is currently not currently supported TwoStageDetector {}'.format(
                self.ymir_model.model.__class__.__name__))
        else:
            raise NotImplementedError('aldd mining is currently not currently supported with {}'.format(
                self.ymir_model.model.__class__.__name__))


def main():
    cfg = get_merged_config()
    miner = ALDDMiner(cfg)
    miner.mining()
    return 0


if __name__ == "__main__":
    sys.exit(main())
