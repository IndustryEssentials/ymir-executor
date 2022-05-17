import os
import os.path as osp
from executor import env
import subprocess
import logging

def main():
    # step 1. read config.yaml and clone git_url:git_branch to /app
    logging.info('step::1 runing git clone ... ','*'*50)
    executor_config = env.get_executor_config()
    git_url=executor_config['git_url']
    git_branch=executor_config['git_branch']

    cmd=f'git clone {git_url} -b {git_branch} -o /app' 
    subprocess.check_output(cmd.split())

    # step 2. read /app/extra-requirements.txt and install it.
    logging.info('step::2 runing pip install ... ','*'*50)
    if osp.exists('/app/extra-requirements.txt'):
        cmd='pip install -r /app/extra-requirements.txt'
        subprocess.check_output(cmd.split())

    # step 3. run /app/start.py 
    logging.info('runing python start.py ','*'*50)
    cmd = 'cd /app && python3 start.py'
    subprocess.check_output(cmd.split())

if __name__ == '__main__':
    main()