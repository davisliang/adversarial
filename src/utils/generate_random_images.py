# This file is for adding random noise to the adversarial images as a form of mitigation
import re
import numpy as np
import os
from os.path import expanduser
from os import listdir
from os.path import isfile
import cv2

def add_noise(image):
    noise = np.random.randint(low=-3, high=4, size=image.shape)
    noisy_image = image + noise
    return noisy_image

def create_noisy_dataset(image_dir, dest_dir):
    dir_path = expanduser("~/adversary/data/") + image_dir
    onlyfiles = [f for f in listdir(dir_path)]
    print (len(onlyfiles))
    print (dir_path)
    for f in onlyfiles:
        imgdata = cv2.imread(dir_path + '/' + f)
        noisy_image = add_noise(imgdata)
        cv2.imwrite(expanduser("~/adversary/data/") + dest_dir +  "/noisy_"
        + str([int(x.group()) for x in re.finditer(r'\d+', f)][0]) + ".JPEG", noisy_image)
