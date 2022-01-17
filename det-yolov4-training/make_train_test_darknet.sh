set -e

cat /in/config.yaml | grep task_id: | tr -s " " "\012" > /out/tmp.txt
echo -e "$(sed '2!d' /out/tmp.txt)\t$(date +%s)000\t"0.00"\t"running"\t" > /out/monitor.txt
# find all jpg image files and save their paths in img.txt
find /in -type f \( -iname \*.jpeg -o -iname \*.png \) > /out/img.txt
find /in -iname "*.jpg" >> /out/img.txt
mkdir -p /out/models

# convert label format from ark format to darknet need to pay attention here the code "convert_label_ark2txt.py" fit for 1 class
python3 ./convert_label_ark2txt.py

find /in/train -type f \( -iname \*.jpeg -o -iname \*.png \) > /in/train.txt
find /in/train -iname *.jpg >> /in/train.txt
find /in/val -type f \( -iname \*.jpeg -o -iname \*.png \) > /in/test.txt
find /in/val -iname *.jpg  >> /in/test.txt

python3 config_and_train.py
echo -e "$(sed '2!d' /out/tmp.txt)\t$(date +%s)000\t"1.0"\t"running"\t" > /out/monitor.txt
