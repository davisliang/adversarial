# This file is for testing the models agaisnt the adversarial images

import numpy
import scipy
from scipy import ndimage
from os.path import expanduser
import tensorflow as tf
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, expanduser('~/adversary/src/models'))
sys.path.insert(0, expanduser('~/adversary/src/utils/Hvass_Lab'))
import inception

def test_on_adv_set():
    inception.maybe_download()
    model = inception.Inception()

    images_control, labels_control = pickle.load(
                open( expanduser("~/adversary/data/control.p"), "rb"))

    images_uni, labels_uni = pickle.load(
                open( expanduser("~/adversary/data/universal.p"), "rb"))

    pred = inception.process_images(fn=model.classify, images=images_control)

    print (pred.shape)

    model.close()

    

