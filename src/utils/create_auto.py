from os import listdir
from os.path import isfile, join
from collections import defaultdict
import re
from shutil import copyfile


id_to_cat = list()
mypath = "/home/phhayes/2012/data"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

filename_to_id = dict()
category_count = defaultdict(int)

with open("/home/phhayes/2012/labels/labels.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
id_to_cat = [x.strip() for x in content] 

for f in onlyfiles:
    filename_to_id[f] = [int(x.group()) for x in re.finditer(r'\d+', f)][1]

with open("/home/phhayes/auto_encoder_trainning_set/labels.txt", 'a') as labels_f:

    i = 1
    j = 0
    for img in onlyfiles:

        cat = id_to_cat[filename_to_id[img]]

        if category_count[cat] < 2:
            #copyfile(join(mypath, img), '/home/phhayes/control/images/control_' + str(i) + '.JPEG')
            #labels_f.write(str(cat) + '\n')
            category_count[cat] += 1
            i += 1
        else:
            copyfile(join(mypath, img), '/home/phhayes/auto_encoder_trainning_set/images/auto_' + str(j) + '.JPEG')
            labels_f.write(str(cat) + '\n')
            j += 1




