import os.path as osp
import subprocess
import sys

from loguru import logger
from ymir_exc import env


def show_ymir_info(executor_config: dict) -> None:
    ymir_env = env.get_current_env()

    logger.info(f'executor config: {executor_config}')
    logger.info(f'ymir input env: {ymir_env.input}')
    logger.info(f'ymir output env: {ymir_env.output}')


def main():
    # step 1. read config.yaml and clone git_url:git_branch to /app
    executor_config = env.get_executor_config()
    show_ymir_info(executor_config)

    git_url = executor_config['git_url']
    git_branch = executor_config.get('git_branch', '')

    if not git_branch:
        cmd = f'git clone {git_url} /app'
    else:
        cmd = f'git clone {git_url} -b {git_branch} /app'
    logger.info(f'clone code: {cmd}')
    subprocess.check_output(cmd.split())

    # step 2. read /app/extra-requirements.txt and install it.
    pypi_file = '/app/extra-requirements.txt'
    if osp.exists(pypi_file):
        pypi_mirror = executor_config.get('pypi_mirror', '')

        cmd = f'pip install -r {pypi_file}'
        cmd += ' -i {pypi_mirror}' if pypi_mirror else ''

        logger.info(f'install python package: {cmd}')
        subprocess.check_output(cmd.split())
    else:
        logger.info('no python package needs to install')

    # step 3. run /app/start.py
    cmd = 'python3 start.py'
    logger.info(f'run task: {cmd}')
    subprocess.check_output(cmd.split(), cwd='/app')

    logger.info('live code executor run successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
