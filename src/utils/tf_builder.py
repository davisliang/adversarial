"""
This utilities script contains functions to easily build tensorflow networks
"""

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

