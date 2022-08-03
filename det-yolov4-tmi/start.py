import logging
import subprocess
import sys

import yaml


def start() -> int:
    with open("/in/env.yaml", "r", encoding='utf8') as f:
        config = yaml.safe_load(f)

    logging.info(f"config is {config}")
    if config['run_training']:
        cmd = 'bash /darknet/make_train_test_darknet.sh'
        cwd = '/darknet'
    else:
        cmd = 'python3 docker_main.py'
        cwd = '/darknet/mining'
    subprocess.run(cmd, check=True, shell=True, cwd=cwd)

    return 0

if __name__ == '__main__':
    sys.exit(start())
