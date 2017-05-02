import tensorflow as tf
import numpy
from numpy import random
from scipy.misc import imresize
import scipy
#from keras.datasets import cifar10
import cPickle as pickle
from os.path import expanduser
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~/adversary/utils'))
import tf_builder
import cifar10_load
import tf_utils

def create_model(x,weights,biases,dropout):
    x = tf.reshape(x, shape=[-1,32,32,3])
    
    conv1 = tf_builder.conv2d(x,weights['conv1_weights'],biases['conv1_biases'])
    conv1 = tf_builder.maxpool2d(conv1,mp_size=3,stride=2)
    conv2 = tf_builder.conv2d(conv1, weights['conv2_weights'], biases['conv2_biases'])
    conv2 = tf_builder.maxpool2d(conv2, mp_size=3,stride=2)

    flattened_conv, weights['fc1_weights'], biases['fc1_biases'] = flatten(conv2,384)
    fc1 = fully_connected(flattened_conv, weights['fc1_weights'],biases['fc1_biases'],dropout=1.0)
    fc2 = fully_connected(fc1, weights['fc2_weights'],biases['fc2_biases'],dropout=1.0)
    out = outputs(fc2,weights['out_weights'],biases['out_biases'])
    return out

def initialize_weights():

    # Store layers weight & bias
    weights = {
        'conv1_weights': tf.Variable(tf.random_normal([5, 5, 3, 64])),
        'conv2_weights': tf.Variable(tf.random_normal([5, 5, 64, 64])),
        'fc2_weights': tf.Variable(tf.random_normal([384,192])),
        'out_weights': tf.Variable(tf.random_normal([192, n_classes]))
    }

    biases = {
        'conv1_biases': tf.Variable(tf.random_normal([64])),
        'conv2_biases': tf.Variable(tf.random_normal([64])),
        'fc2_biases': tf.Variable(tf.random_normal([192])),
        'out_biases': tf.Variable(tf.random_normal([n_classes]))
    }
    
    return weights, biases


def run_model():
    print "starting script"
    (x_train_,y_train_), (x_test, y_test) = cifar10_load.load_cifar()
    print "data loaded"

    print 'creating model'
    x = tf.placeholder(tf.float32, [None, height, width, channels])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    weights, biases = initialize_weights()
    pred = create_model(x, weights, biases, keep_prob)
    softmax_pred = tf.nn.softmax(pred)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    for weight in weights.values():
        cost = cost + regularization*tf.nn.l2_loss(weight)
    
    annealed_rate = tf.train.exponential_decay(learning_rate, tf.Variable(0, trainable=False), 100000, .96, staircase=True)
    optimizer = tf.train.AdamOptimizer(annealed_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    print 'training'
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            step = 1
            x_train, y_train = tf_utils.shuffle_data(x_train_, y_train_)

            
            # Keep training until reach max iterations
            training_iters = len(x_train)
            while step * batch_size < training_iters:
                batch_x, batch_y = x_train[(step-1)*batch_size:step*batch_size], y_train[(step-1)*batch_size:step*batch_size]

                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                               keep_prob: dropout})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc, pred_val = sess.run([cost, accuracy, softmax_pred], feed_dict={x: batch_x,
                                                                      y: batch_y,
                                                                      keep_prob: 1.})
                
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for 256 mnist test images
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: x_test,
                                              y: y_test,
                                              keep_prob: 1.}))   

height = 32
width = 32
channels = 3
n_classes = 10
batch_size = 1024
dropout = 1.0
learning_rate = 0.00001
display_step = 10
epochs = 500
regularization = 0.00


setup = tf_builder.setup('params.csv')
run_model()
