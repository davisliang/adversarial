import tensorflow as tf
import numpy
from scipy.misc import imresize
import scipy
import cPickle as pickle
from os.path import expanduser
import sys
sys.path.insert(0, expanduser('~/adversary/src/utils'))
import tf_builder
import cifar10_load
import tf_utils

"""
building the computational graph of the model
"""
def create_model(x,weights,biases,dropout):
    x = tf.reshape(x, shape=[-1,32,32,3])
    
    conv1 = tf_builder.conv2d(x,weights['conv1_weights'],biases['conv1_biases'])
    pool1 = tf_builder.maxpool2d(conv1,mp_size=3,stride=2)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')


    conv2 = tf_builder.conv2d(norm1, weights['conv2_weights'], biases['conv2_biases'])
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf_builder.maxpool2d(norm2, mp_size=3,stride=2)

    flattened_conv, weights['fc1_weights'], biases['fc1_biases'] = tf_builder.flatten(pool2,int(param_dict['fc2_weights'][0]))
    fc1 = tf_builder.fully_connected(flattened_conv, weights['fc1_weights'],biases['fc1_biases'],dropout=1.0)
    fc2 = tf_builder.fully_connected(fc1, weights['fc2_weights'],biases['fc2_biases'],dropout=1.0)
    out = tf_builder.output(fc2,weights['out_weights'],biases['out_biases'])
    return out

"""
training and testing model
"""
def run_model(param_dict):

    #initializing parameters
    weights, biases, height, width, channels, n_classes, batch_size, \
        dropout, learning_rate, display_step, epochs, regularization = tf_builder.initialize_parameters(param_dict)

    #loading data
    (x_train_,y_train_), (x_test, y_test) = cifar10_load.load_cifar()

    #creating input/output placeholders
    x = tf.placeholder(tf.float32, [None, height, width, channels])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    #create model
    pred = create_model(x, weights, biases, keep_prob)
    softmax_pred = tf.nn.softmax(pred)

    #computing cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

    #l2 regularization
    #for weight in weights.values():
    #    cost = cost + regularization*tf.nn.l2_loss(weight)
    
    #training model
    annealed_rate = tf.train.exponential_decay(learning_rate, tf.contrib.framework.get_or_create_global_step(), epochs*350/batch_size, 0.1, staircase=True)
    optimizer = tf.train.AdamOptimizer(annealed_rate).minimize(cost)

    #evaluating model
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #initializing the variables
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

param_dict = tf_builder.setup('~/adversary/data/param.csv')
run_model(param_dict)
