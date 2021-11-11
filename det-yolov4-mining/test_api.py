from active_learning import ALAPI

from tqdm import tqdm
import threading


api = ALAPI(
    selected_img_list_path="al_select.txt",
    unlabeled_img_list_path="train_data_path/test_api.txt",
    labeled_img_list_path="labeled.txt",
    strategy="aldd", proportion=None, absolute_number=5000,
    model_type="detection", model_name="centernet",
    model_params_path="/home/xiongzihua/code/centernet-mx/deploy_model/mobilenet_sc_cpu_combined_random_select_5000-0130.params", gpu_id='7'
)

print(api.is_completed, api.progress)
t = threading.Thread(target=api.run)
t.start()
pre = 0
with tqdm(total=100, desc='Progress', ncols=100, ascii=' =') as pbar:
    while t.is_alive():
        if int(api.progress * 100) > pre:
            pbar.update(int(api.progress * 100) - pre)
            pre = int(api.progress * 100)
t.join()
print(api.is_completed, api.progress)
print("done")