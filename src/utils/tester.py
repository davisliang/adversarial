# This file is for testing the models agaisnt the adversarial images

import numpy
import scipy
from scipy import ndimage
from os.path import expanduser
import tensorflow as tf
from PIL import Image
import pickle
import os
import sys 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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

    print (images_uni.shape)
    print (labels_uni[:10])

    pred = inception.process_images(fn=model.classify, images=images_uni)
    
    #picks the category with the highest score
    labels_true = numpy.argmax(labels_uni, axis=1)
    labels_pred = numpy.argmax(pred, axis=1)

    print (labels_pred[:10])
    
    conf_matrix = confusion_matrix(labels_true, labels_pred) 
    
    accuracy = accuracy_score(labels_true, labels_pred)

    print ("Accuracy = " + str(accuracy) + '/n')
    print ("Confusion Matrix: " + str(conf_matrix) + '/n')

    model.close()

    

