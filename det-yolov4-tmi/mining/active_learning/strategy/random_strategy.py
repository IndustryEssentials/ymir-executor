import random


def random_select(unlabeled_dataset, sampled_num):
    img_list = random.sample(unlabeled_dataset.img_list, sampled_num)
    return img_list