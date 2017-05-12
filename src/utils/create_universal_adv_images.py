from os import listdir
from os.path import isfile, join
from collections import defaultdict
import re
from shutil import copyfile
import numpy
import scipy
import tensorflow as tf
from scipy.io import loadmat
import os
from os.path import expanduser
from random import randint
from scipy import misc
import cv2

mypath = "/home/phhayes/control/images"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
path_to_dir = "/home/phhayes/adversary/data/"

def get_precomputed_universal_perturbations(path_to_dir):
    universal=[]
    for filename in os.listdir(expanduser(path_to_dir)):
        if filename[-4:] == '.mat':
            universal.append(loadmat(expanduser(path_to_dir+str(filename)))['r'])
    return universal

def generate_universal_adversarial_images(data, perturbation, imshape):
    if(len(imshape)!= 3): return 
    perturbed_image = numpy.zeros((imshape[0],imshape[1],imshape[2]))
    currImage = scipy.misc.imresize(data[:,:,:],imshape)
    perturbed_image[:,:,:] = perturbation*1.0 + currImage*1.0
    return perturbed_image

universal_perturbations = get_precomputed_universal_perturbations(path_to_dir)

for f in onlyfiles:
    imgdata =cv2.imread(join(mypath,f))
    advimg = generate_universal_adversarial_images(imgdata, universal_perturbations[randint(0,5)], (224,224,3))
    cv2.imwrite("/home/phhayes/universal_adv/images/universal_"
            + str([int(x.group()) for x in re.finditer(r'\d+', f)][0]) + ".JPEG", advimg)

