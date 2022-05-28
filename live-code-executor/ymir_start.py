import os.path as osp
import subprocess
import sys

from loguru import logger
from ymir_exc import env


def main():
    # step 1. read config.yaml and clone git_url:git_branch to /app
    executor_config = env.get_executor_config()

    git_url = executor_config['git_url']
    git_branch = executor_config['git_branch']

    cmd = f'git clone {git_url} -b {git_branch} /app'
    logger.info(f'step::1 {cmd} '+'*'*20)
    subprocess.check_output(cmd.split())

    # step 2. read /app/extra-requirements.txt and install it.
    pypi_file = '/app/extra-requirements.txt'
    if osp.exists(pypi_file):
        pypi_mirror = executor_config.get(
            'pypi_mirror', 'https://pypi.tuna.tsinghua.edu.cn/simple')
        if not pypi_mirror:
            cmd = f'pip install -r {pypi_file}'
        else:
            cmd = f'pip install -r {pypi_file} -i {pypi_mirror}'

        logger.info(f'step::2 {cmd} '+'*'*20)
        subprocess.check_output(cmd.split())
    else:
        logger.info(f'step::2 not find {pypi_file} '+'*'*20)

    # step 3. run /app/start.py
    cmd = 'python3 start.py'
    logger.info(f'step::3 {cmd} '+'*'*20)
    subprocess.check_output(cmd.split(), cwd='/app')
    return 0


if __name__ == '__main__':
    sys.exit(main())
