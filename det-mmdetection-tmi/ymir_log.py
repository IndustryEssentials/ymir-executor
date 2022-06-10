import time
import os.path as osp
from typing import Generator
from pygtail import Pygtail
from mmcv.util import TORCH_VERSION, digit_version

if (TORCH_VERSION == 'parrots'
        or digit_version(TORCH_VERSION) < digit_version('1.1')):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError('Please install tensorboardX to use '
                          'TensorboardLoggerHook.')
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError(
            'Please run "pip install future tensorboard" to install '
            'the dependencies to use torch.utils.tensorboard '
            '(applicable to PyTorch 1.1 or higher)')


def read_log(f: str, wait: bool = True, sleep: float = 0.1) -> Generator[str]:
    """
    Basically tail -f with a configurable sleep
    """
    with open(f) as logfile:
        # logfile.seek(0, os.SEEK_END)
        while True:
            new_line = logfile.readline()
            if new_line:
                yield new_line
            else:
                if wait:
                    # wait for new line
                    time.sleep(sleep)
                else:
                    # read all line in file
                    break

def write_tensorboard_text(tb_log_file: str, executor_log_file: str) -> None:
    global _TENSORBOARD_GLOBAL_STEP
    # tb_log_file = osp.join(cfg.ymir.output.tensorboard_dir, 'tensorboard_text.log')
    # executor_log_file = cfg.ymir.output.executor_log_file
    writer = SummaryWriter(tb_log_file)

    # Pygtail always return the new lines
    for line in Pygtail(executor_log_file):
        writer.add_text(tag='ymir-executor', text_string=line, global_step=_TENSORBOARD_GLOBAL_STEP)
        _TENSORBOARD_GLOBAL_STEP += 1

    writer.close()