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

def test_dataset(pickle_f):
    inception.maybe_download()
    model = inception.Inception()

    images, labels = pickle.load(
                open( expanduser("~/adversary/data/pickles/" + pickle_f), "rb"))

    pred = inception.process_images(fn=model.classify, images=images)
    
    #picks the category with the highest score
    labels_true = numpy.argmax(labels, axis=1)
    labels_pred = numpy.argmax(pred, axis=1)
    
    conf_matrix = confusion_matrix(labels_true, labels_pred) 
    
    accuracy = accuracy_score(labels_true, labels_pred)
    
    f = open(expanduser("~/adversary/data/results/" + pickle_f), 'w')
    f.write("Accuracy =  " + str(accuracy) + '\n')  # python will convert \n to os.linesep
    f.write("Confusion Matrix: " + str(conf_matrix) + '\n')
    f.close()

    model.close()


