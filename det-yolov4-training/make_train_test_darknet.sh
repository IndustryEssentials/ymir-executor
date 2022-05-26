set -e

in="${1:-/in}"
out="${2:-/out}"

mkdir -p ${out}/models

# update monitor, state: running
cat ${in}/config.yaml | grep task_id: | tr -s " " "\012" > ${out}/tmp.txt
echo -e "$(sed '2!d' ${out}/tmp.txt)\t$(date +%s).000000\t"0.00"\t"2"\t" > ${out}/monitor.txt

# convert label format from ark format to darknet need to pay attention here the code "convert_label_ark2txt.py" fit for 1 class
python3 ./convert_label_ark2txt.py

# start training
# training set: /in/train/index.tsv, annotations: /in/train/*.txt
# validation set: /in/val/index.tsv, annotations: /in/val/*.txt
python3 config_and_train.py

# update monitor, state: running
echo -e "$(sed '2!d' ${out}/tmp.txt)\t$(date +%s).000000\t"1.0"\t"2"\t" > ${out}/monitor.txt
