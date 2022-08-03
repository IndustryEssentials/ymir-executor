import os
import mxnet 
import argparse

def warmup_training(train_script, gpus):
    gpus = gpus.split(",")
    if len(gpus) == 1:
        os.system(train_script)
    else:
        unused_gpus = gpus[1:]
        occupy_ctx = [mxnet.gpu(int(each_gpuid)) for each_gpuid in unused_gpus]
        occupy_list = []
        for each_ctx in occupy_ctx:
            each_occupy_content = mxnet.ndarray.ones([1, 3, 20000, 25000], ctx=each_ctx)
            occupy_list.append(each_occupy_content)

        os.system(train_script)
        del occupy_list
        for each_ctx in occupy_ctx:
            each_ctx.empty_cache() 


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='input class info and netwrok input size')
    parser.add_argument('--train_script', type=str, default="", help='train script of warmup training')
    parser.add_argument('--gpus', type=str, default="", help='gpus that use in warmup process')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    warmup_training(args.train_script, args.gpus)
