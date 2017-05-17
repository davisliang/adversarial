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
import inception

def test_on_adv_set():
    inception.maybe_download()
    model = inception.Inception()

    images_control, labels_control, images_uni, labels_uni = pickle.load(
                open( expanduser("~/adversary/data/control_and_universal.pickle", "rb")))

    

