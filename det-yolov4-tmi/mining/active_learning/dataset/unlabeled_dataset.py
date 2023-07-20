import shutil


class UnlabeledDataset:
    def __init__(self, img_list_path, dst_img_list_path=None):
        self.img_list_path = img_list_path
        self.dst_img_list_path = dst_img_list_path
        with open(self.img_list_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_samples = len(self.img_list)

        # # backup
        # prefix, suffix = self.img_list_path.split('.')
        # backup_path = prefix + "_al_backup." + suffix
        # shutil.copyfile(self.img_list_path, backup_path)

    def delete(self, minus_img_list):
        #  update(minus) unlabeled dataset
        self.img_list = list(set(self.img_list) - set(minus_img_list))
        with open(self.dst_img_list_path, 'w') as f:
            f.writelines(self.img_list)