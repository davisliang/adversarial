"""
This utilities script contains functions to easily build tensorflow networks
"""
import tensorflow as tf
import csv 
import numpy
from os.path import expanduser
"""
conv2d
"""
def conv2d(inputs, weight, bias, stride=1, pad_type='SAME'):
	val = tf.nn.conv2d(inputs, weight, strides=[1,stride,stride,1], padding=pad_type)
	val = tf.nn.bias_add(val,bias)
	return tf.nn.relu(val)

"""
maxpooling
"""
def maxpool2d(inputs,mp_size,stride):
	return tf.nn.max_pool(inputs,ksize=[1,mp_size,mp_size,1],strides=[1,stride,stride,1],padding='SAME')

"""
flatten
"""
def flatten(prev_layer,next_size):
	next_weight = initialize_flatten_weights(prev_layer.get_shape().as_list(), next_size)
	next_bias = tf.Variable(tf.random_normal([next_size]))
	flattened_input = tf.reshape(prev_layer, [-1, next_weight.get_shape().as_list()[0]])
	return flattened_input, next_weight, next_bias
"""
flatten helper function
"""
def initialize_flatten_weights(input_size, output_size):
	total_size = 1
	for i in range(len(input_size)-1):
		total_size = total_size * input_size[i+1]
	return tf.Variable(tf.random_normal([total_size, output_size]))

"""
fully connected
"""
def fully_connected(inputs, weight, bias, dropout):
	net = tf.add(tf.matmul(inputs, weight), bias)
	out = tf.nn.relu(net)
	out = tf.nn.dropout(out, dropout)
	return out

"""
output
"""
def output(inputs,weight,bias):
	out = tf.add(tf.matmul(inputs, weight), bias)
	return out

"""
output with softmax
"""
def output_softmax(inputs, weight, bias):
	out = tf.add(tf.matmul(inputs,weight,bias))
	softmax_out = tf.nn.softmax(out)
	return softmax_out

"""
return dictionary of params given params.csv file
"""
def setup(params_file):
	file = open(expanduser(params_file))
	csvfile = csv.reader(file)

	dictionary = {}
	for row in csvfile:
		if(row == []):
			continue
		key = row[0]
		value = row[1:]
		value = [float(string) for string in value]
		dictionary[key]=value
	return dictionary

"""
modifiable code for loading parameters from file
"""
def initialize_parameters(param_dict):

	#store parameters
	height = int(param_dict['height'][0])
	width = int(param_dict['width'][0])
	channels = int(param_dict['channels'][0])
	n_classes = int(param_dict['n_classes'][0])
	batch_size = int(param_dict['batch_size'][0])
	dropout = param_dict['dropout'][0]
	learning_rate = param_dict['learning_rate'][0]
	display_step = int(param_dict['display_step'][0])
	epochs = int(param_dict['epochs'][0])
	regularization = param_dict['regularization'][0]

	# Store layers weight & bias
	weights = {}
	biases = {}
	
	#use the weights and biases in the for loops
	for key in param_dict:
		if(key[-7:] == 'weights'):
			weights[key] = tf.Variable(tf.random_normal(numpy.asarray(param_dict[key],dtype='int32')))
		if(key[-6:] == 'biases'):
			biases[key] = tf.Variable(tf.random_normal(numpy.asarray(param_dict[key],dtype='int32')))

	return weights, biases, height, width, channels, n_classes, batch_size, \
		dropout, learning_rate, display_step, epochs, regularization