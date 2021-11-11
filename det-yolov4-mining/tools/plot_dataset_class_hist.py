from collections import defaultdict
from tqdm import tqdm

with open("./train_data_path/labeled_random_5w.txt", 'r') as f:
    lines = f.readlines()

# with open("./temp/cald_5w_add_1w_score.txt", 'r') as f:
#     lines = f.readlines()
# lines = [x.split(' ')[0] for x in lines]

# base = lines[0:5000]
# select = lines[5000:]
select = lines
# base = [x.strip() for x in base]
select = [x.strip() for x in select]

# base_dataset_hist = defaultdict(int)
select_dataset_hist = defaultdict(int)


def count_dataset(paths, hist):
    for path in paths:
        data_name = path.split('/')[6]
        hist[data_name] += 1
    for k, v in hist.items():
        hist[k] = round(v / float(len(paths)), 4)


classes = ["bike", "car", "face", "license", "person"]
# base_classes_hist = defaultdict(int)
select_classes_hist = defaultdict(int)


def count_classes(paths, hist):
    for path in tqdm(paths):
        path = path[:-4] + ".txt"
        # print(path)
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls = int(line.split(' ')[0])
                cls = classes[cls]
                hist[cls] += 1
    total = 0
    for k, v in hist.items():
        total += v
    for k, v in hist.items():
        hist[k] = (round(v / float(len(paths)), 4), round(v / float(total), 4))


# count_dataset(base,  base_dataset_hist)
count_dataset(select, select_dataset_hist)
# print(base_dataset_hist)
print(select_dataset_hist)

# count_classes(base, base_classes_hist)
count_classes(select, select_classes_hist)
# print(base_classes_hist)
print(select_classes_hist)
