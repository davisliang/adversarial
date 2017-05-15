
from keras.regularizers import l2
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from os.path import expanduser
from keras.applications import VGG16
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator
import numpy
import keras
from keras.models import Model
import scipy
from keras.datasets import cifar100
import utils

print "loading shuffled raw train, validation, and test images"
block = pickle.load(open(expanduser("~/Desktop/Crystal/crystal_data.pkl")))
[trainData,trainLabel,validData,validLabel,_,_] = block
sampling = False

img_width, img_height = 224, 224

print "compiling network"

model = VGG16(weights='imagenet',include_top=True)
vgg_body = model.layers[-1].output
softmax_layer = keras.layers.core.Dense(4,W_regularizer=l2(0.01),init='glorot_uniform',activation='softmax')(vgg_body)
tl_model = Model(input=model.input, output=softmax_layer)


for layer in model.layers:
    layer.trainable = False

tl_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Nadam(),
              metrics=['accuracy'])

tl_model.summary()

numEpochs=100


train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)


if(sampling == False):
  tr_sample_data, tr_sample_label = utils.listOfImagesToNumpy(trainData, trainLabel)
  va_sample_data, va_sample_label = utils.listOfImagesToNumpy(validData, validLabel)
else: 
  tr_sample_data, tr_sample_label, tr_sample_name = utils.sampleSet(trainData,trainLabel,None,10)
  va_sample_data, va_sample_label, tr_sample_name = utils.sampleSet(validData,validLabel,None,10)

tr_sample_data = numpy.transpose(tr_sample_data,[0,2,3,1])
va_sample_data = numpy.transpose(va_sample_data,[0,2,3,1])

train_datagen.fit(tr_sample_data)
valid_datagen.fit(va_sample_data)

#tl_model.fit_generator(train_datagen.flow(tr_sample_data, tr_sample_label, batch_size=32), 
#        samples_per_epoch=len(tr_sample_data), nb_epoch=numEpochs,
#        validation_data=valid_datagen.flow(va_sample_data,va_sample_label, batch_size=32),
#        nb_val_samples=len(va_sample_data))

weightings = {0:1.0,1:1.9888268,2:4.0454545,3:3.67010309}
tl_model.fit_generator(train_datagen.flow(tr_sample_data, tr_sample_label, batch_size=32), 
        samples_per_epoch=len(tr_sample_data), nb_epoch=numEpochs, class_weight = weightings,
        validation_data=valid_datagen.flow(va_sample_data,va_sample_label, batch_size=32),
        nb_val_samples=len(va_sample_data))


model_json = tl_model.to_json()
with open("model_after_first_run_crystal_four.json","w") as json_file:
    json_file.write(model_json)

tl_model.save_weights("model_after_first_run_crystal_four.h5")

print "saved"

del tr_sample_data
del va_sample_data
del tr_sample_label
del va_sample_label
del valid_datagen
del train_datagen
