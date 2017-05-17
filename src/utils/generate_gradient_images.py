# This file is for generating the adversarial using gradient ascent.
# We take the gradient of the loss functions acording to some target
# class (like "cat" or "frog") and increase the pixel values acording
# to the negative gradient

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import gradient_utils as gutils
from os.path import expanduser

# Functions and classes for loading and using the Inception model.
import inception

# specify the location where we should keep all the files
# associated with the inception model
inception.data_dir = expanduser("~/adversary/src/models/inception")

#Download the data for the Inception model if it doesn't already
# exist in the directory. It is 85 MB.
inception.maybe_download()

#Load the Inception model so it is ready for classifying images.
model = inception.Inception()

#Get a reference to the input tensor for the Inception model.
resized_image = model.resized_image

#Get a reference to the output of the softmax-classifier for the Inception model.
y_pred = model.y_pred

#Get a reference to the unscaled output of the softmax-classifier for the Inception
# model. These are often called 'logits'. The logits are necessary because we will
# add a new loss-function to the graph, which requires these unscaled outputs.
y_logits = model.y_logits


# Set the graph for the Inception model as the default graph,
# so that all changes inside this with-block are done to that graph.
with model.graph.as_default():
    # Add a placeholder variable for the target class-number.
    # This will be set to e.g. 300 for the 'bookcase' class.
    pl_cls_target = tf.placeholder(dtype=tf.int32)

    # Add a new loss-function. This is the cross-entropy.
    # See Tutorial #01 for an explanation of cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])

    # Get the gradient for the loss-function with regard to
    # the resized input image.
    gradient = tf.gradients(loss, resized_image)


# We need a Tensorflow session to execute the graph
session = tf.Session(graph=model.graph)

gutils.generate_adversarial_dataset("~/adversary/data/images_control",
                                     "~/adversary/data/labels_control.txt",
                                     "~/adversary/data/images_control_gradient",
                                     session, y_pred, resized_image, gradient,
                                     pl_cls_target)
