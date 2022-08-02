# python al_main.py --extracted_num 2000 --unlabeled_img_list_path train_data_path/unlabeled_base.txt --labeled_img_list_path train_data_path/combined_al_train_base.txt --dst_unlabeled_img_list_path train_data_path/random_select_unlabel_iter1_2000.txt --dst_labeled_img_list_path train_data_path/random_select_labeled_iter1_2000.txt
# python al_main.py --absolute_num 2000 --unlabeled_img_list_path /home/xiongzihua/code/active_learning/train_data_path/aldd_select_unlabeled_iter2_2000.txt --labeled_img_list_path /home/xiongzihua/code/active_learning/train_data_path/aldd_select_labeled_iter2_2000.txt --dst_unlabeled_img_list_path /home/xiongzihua/code/active_learning/train_data_path/temp_aldd_select_unlabeled_iter3_2000.txt --dst_labeled_img_list_path /home/xiongzihua/code/active_learning/train_data_path/temp_aldd_select_labeled_iter3_2000.txt --strategy aldd --model_params_path ../centernet-mx/deploy_model/mobilenet_sc_cpu_combined_aldd_select_iter2_2000-0130.params --model_name centernet --gpu_id 1

import argparse
import time
import threading
from tqdm import tqdm

from active_learning import ALAPI


def arg_parse():
    parser = argparse.ArgumentParser(description = "Active Learning")
    parser.add_argument("--strategy", dest = "strategy", help = "random or others", default = "random",
                        type = str)
    parser.add_argument("--proportion", dest = 'proportion', help = "proportion of data extracted from unlabeled dataset",
                        default = None, type = int)
    parser.add_argument('--absolute_number', dest = 'absolute_number', help = "number of data extracted from unlabeled dataset",
                        default = None, type = int)
    parser.add_argument("--unlabeled_img_list_path", dest = 'unlabeled_img_list_path', type = str)
    parser.add_argument("--labeled_img_list_path", dest = 'labeled_img_list_path', type = str)
    parser.add_argument("--dst_unlabeled_img_list_path", dest = 'dst_unlabeled_img_list_path', type = str)
    parser.add_argument("--dst_labeled_img_list_path", dest = 'dst_labeled_img_list_path', type = str)

    parser.add_argument("--model_params_path", dest = 'model_params_path', type = str)
    parser.add_argument("--model_type", dest = 'model_type', type = str)
    parser.add_argument("--model_name", dest = 'model_name', type = str)
    parser.add_argument("--gpu_id", dest = 'gpu_id', default='0', type = str)
    parser.add_argument("--task_id", dest = 'task_id', default='al', type = str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    unlabeled_img_list_path = args.unlabeled_img_list_path
    labeled_img_list_path = args.labeled_img_list_path
    dst_unlabeled_img_list_path = args.dst_unlabeled_img_list_path
    dst_labeled_img_list_path = args.dst_labeled_img_list_path
    strategy = args.strategy
    proportion = args.proportion
    absolute_number = args.absolute_number
    model_params_path = args.model_params_path
    model_type = args.model_type
    model_name = args.model_name
    gpu_id = args.gpu_id
    task_id = args.task_id

    api = ALAPI(
        selected_img_list_path="./temp/{}_{}_result.txt".format(strategy, task_id),
        unlabeled_img_list_path=unlabeled_img_list_path,
        labeled_img_list_path=labeled_img_list_path,
        dst_unlabeled_img_list_path=dst_unlabeled_img_list_path,
        dst_labeled_img_list_path=dst_labeled_img_list_path,
        strategy=strategy, proportion=proportion, absolute_number=absolute_number,
        model_type=model_type, model_name=model_name, model_params_path=model_params_path, gpu_id=gpu_id, task_id=task_id
    )

    # run select
    t = threading.Thread(target=api.run)
    t.start()
    pre = 0
    total = 100
    with tqdm(total=total, desc='Progress', ncols=total, ascii=' =') as pbar:
        while t.is_alive():
            if int(api.progress * total) > pre:
                pbar.update(int(api.progress * total) - pre)
                pre = int(api.progress * total)
    t.join()

    # merge labeled data & delete unlabeled data
    api.labeled_dataset.merge(api.selected_img_list)
    api.unlabeled_dataset.delete(api.selected_img_list)
    print("done")
