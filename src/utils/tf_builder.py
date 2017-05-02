"""
This utilities script contains functions to easily build tensorflow networks
"""
import csv 

"""
conv2d
"""
def conv2d(inputs, weight, bias, stride=1, pad_type='SAME'):
    val = tf.nn.conv2d(inputs, weight, strides=[1,stride,stride,1], padding=pad_type)
    val = tf.nn.bias_add(val,b)
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
	net = tf.add(tf.matmul(inputs, weight, bias))
    out = tf.nn.relu(net)
    out = tf.nn.dropout(out, dropout)
    return out

"""
output
"""
def output(inputs,weight,bias):
	out = tf.add(tf.matmul(inputs, weight, bias))
    return out

"""
return dictionary of params given params.csv file
"""
def setup(params_file):
	file = open(expanduser(params_file))
	csvfile = csv.reader(file)

	dictionary = {}
	for row in csvfile:
	    key = row[0]
	    value = row[1:]
	    value = [float(string) for string in value]
	    dictionary[key]=value
	return dictionary