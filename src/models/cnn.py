import tensorflow as tf
import numpy
from numpy import random
from scipy.misc import imresize
import scipy
#from keras.datasets import cifar10
import cPickle as pickle
from os.path import expanduser

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k,s):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,s,s,1],padding='SAME')

def create_model(x,weights,biases,dropout):
    x = tf.reshape(x, shape=[-1,32,32,3])
    
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,k=3,s=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=3,s=2)

    #conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv3 = maxpool2d(conv3, k=2,s=2)

    #conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    #conv4 = maxpool2d(conv4, k=2,s=2)
    
    print "Conv1: ", conv1.get_shape()
    print "Conv2: ", conv2.get_shape()
    #print "Conv3: ", conv3.get_shape()
    #print "Conv4: ", conv4.get_shape()
    weights['wd1'] = initialize_flatten_weights(conv2.get_shape().as_list(), 384)
    
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    print fc1.get_shape().as_list()
    fc2 = tf.add(tf.matmul(fc1, weights['fc2']),biases['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2,dropout)
    

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

def initialize_flatten_weights(input_size, output_size):
    total_size = 1
    for i in range(len(input_size)-1):
        total_size = total_size * input_size[i+1]
    return tf.Variable(tf.random_normal([total_size, output_size]))

def initialize_weights():

    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 64])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 64])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'fc2': tf.Variable(tf.random_normal([384,192])),
        'out': tf.Variable(tf.random_normal([192, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([384])),
        'fc2': tf.Variable(tf.random_normal([192])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    return weights, biases

def download_cifar():

    print 'initializing dataset (CIFAR-10) download script'
    reshape_height = 32
    reshape_width = 32
    reshape_channels = 3
    
    print 'getting dataset (CIFAR-10) from pickled data (50000,32,32,3 ... 10000,32,32,3)'
    x= pickle.load(open(expanduser("~/adversary/data/cifar_data.pkl"),"r"))
    (X_train, y_train), (X_test, y_test) = x
        
    X_train_new = numpy.zeros((50000,reshape_height,reshape_width,reshape_channels))
    X_test_new = numpy.zeros((10000,reshape_height,reshape_width,reshape_channels))
        
    for i in range(len(X_train)):
        if i % 100 == 0:
            print "upsampling training image " + str(i)
        X_train_new[i,:,:,:] = scipy.misc.imresize(X_train[i,:,:,:], [reshape_height,reshape_width,reshape_channels])
    
    for i in range(len(X_test)):
        if i % 100 == 0:
            print "upsampling testing image " + str(i)
        X_test_new[i,:,:,:] = scipy.misc.imresize(X_test[i,:,:,:], [reshape_height,reshape_width,reshape_channels])

    #normalizing data
    X_train_final = X_train_new/255
    X_test_final = X_test_new/255

    #mean = numpy.zeros((32,32,3))
    #for i in range(len(X_train)):
    #    mean = mean + X_train[i,:,:,:]
    #mean /= len(X_train)

    #std = numpy.zeros((32,32,3))
    #for i in range(len(X_train)):
    #    std = std+ numpy.square(X_train[i,:,:,:] - mean)
    #std /= len(X_train-1)

    #X_train = numpy.divide((X_train-mean),std)
    #X_test = numpy.divide((X_test-mean),std)

    x = [(X_train_final, y_train), (X_test_final, y_test)]
    return x
    #pickle.dump(x, open(expanduser("~/adversary/data/cifar_data.pkl"),"w"))
    
def load_cifar():
    x= pickle.load(open(expanduser("~/adversary/data/cifar_data.pkl"),"r"))
    return x

def shuffle_data(x, y):
    shuf_idx = range(len(x))
    random.shuffle(shuf_idx)
    shuf_x = numpy.zeros(x.shape)
    shuf_y = numpy.zeros(y.shape)
    for i, idx in enumerate(shuf_idx):
        shuf_x[i,:,:,:] = x[idx,:,:,:]
        shuf_y[i,:] = y[idx,:]
    return shuf_x, shuf_y

def run_model():
    print "starting script"
    (x_train_,y_train_), (x_test, y_test) = download_cifar()
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
            x_train, y_train = shuffle_data(x_train_, y_train_)

            
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
run_model()
