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
    # commit id, tag or branch
    git_revision = executor_config.get('git_branch', '')

    # https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/
    cmd = f'git clone --filter=blob:none {git_url} /app'
    logger.info(f'running: {cmd}')
    subprocess.run(cmd.split(), check=True)

    if not git_revision:
        # logger.warning(f'no commid_id/tag/branch offered for {git_url}')
        raise Exception(f'no commid_id/tag/branch offered for {git_url}')

    cmd = f'git checkout {git_revision}'
    logger.info(f'running: {cmd}')
    subprocess.run(cmd.split(), check=True, cwd='/app')

    # step 2. read /app/extra-requirements.txt and install it.
    pypi_file = '/app/extra-requirements.txt'
    if osp.exists(pypi_file):
        pypi_mirror = executor_config.get('pypi_mirror', '')

        cmd = f'pip install -r {pypi_file}'
        cmd += ' -i {pypi_mirror}' if pypi_mirror else ''

        logger.info(f'install python package: {cmd}')
        subprocess.run(cmd.split(), check=True)
    else:
        logger.info('no python package needs to install')

    # step 3. run /app/start.py
    cmd = 'python3 start.py'
    logger.info(f'run task: {cmd}')
    subprocess.run(cmd.split(), check=True, cwd='/app')

    logger.info('live code executor run successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
