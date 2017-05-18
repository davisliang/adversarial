''' 
this conda script will call other scripts to do the follow tasks

(1) have a function to load the control test set images
(2) have a function to load the universally perturbed test set images
(3) have a function to grab the other holdout training images
(4) train an autoencoder with the holdout training images
(5) have a function for testing a network on any input and output

'''

import numpy
import scipy
from scipy import ndimage
from os.path import expanduser
from PIL import Image
import pickle
import os
from numpy import random
import re




# Converets the text to an int if it is an number
def atoi(text):
	return int(text) if text.isdigit() else text

# A method that can be passed into sort to allow files
# with numbers in their names to be sorted as numbers
# instead of characters. For eaxmple 1,2,10 instead of
# 1,10,2.
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text) ]

# Lists all non hidden files in the given directory
def listdir_ordered(path):
    lst = os.listdir(path)
    lst.sort(key=natural_keys)
    for f in lst:
        if not f.startswith('.'):
            yield f

def extractLabels(filename):
    labels = open(expanduser(filename));
    labels_list = [];
    for label in labels:
        labels_list.append(int(label[:-1])-1);
    return labels_list

def extractData(directoryname):
    
    # size of image dataset
    num_images = len(list(listdir_ordered(expanduser(directoryname))))
    # return numpy image array
    images = []
    images_names = []
    
    # for each image in directory, grab and crop and save to pickle
    for imagefile in list(listdir_ordered(expanduser(directoryname))):
        if(imagefile[-4:] == 'JPEG' or imagefile[-4:] == 'jpeg'):
            image = Image.open(expanduser(directoryname + '/' + imagefile))
            if(image.mode == 'RGB'):
                images.append(image.copy())
                images_names.append(imagefile)
            else:
                images.append(image.convert('RGB').copy())
                images_names.append(imagefile)
            image.close()
    
    return images, images_names

def resizeSet(images):
    resizedImages = [];
    for image in images:
        resizedImages.append(scipy.misc.imresize(image,(224,224,3), interp='bilinear'))
    return resizedImages

def listOfImagesToNumpy(imageList, labelList):
    returnArray = numpy.zeros((len(imageList),224,224, 3))
    returnLabels = numpy.zeros((len(labelList),1000))
    
    for i in range(len(imageList)):
        returnArray[i,:,:,:]=numpy.reshape(scipy.misc.imresize(numpy.asarray(imageList[i]),(224,224,3),
            interp='bilinear'), (224,224,3))
        returnLabels[i,labelList[i]] = 1
    
    
    return returnArray, returnLabels

def pickleDataset(image_dir, labels_f, pickle_name):
    images_path = expanduser("~/adversary/data/images/" + image_dir)
    label_path = expanduser("~/adversary/data/labels/" + labels_f)
    
    image_list, name_list = extractData(images_path)
    label_list = extractLabels(label_path)
    images = resizeSet(image_list)
    
    images_final, labels_final = listOfImagesToNumpy(images, label_list)

    tup = [images_final, labels_final]

    pickle.dump( tup, open(expanduser(~/adversary/data/pickles/ + '/' + pickle_name), "wb" ) )
    return
