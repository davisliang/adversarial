"""
tensorflow data processing utilities
"""

"""
shuffle data
"""
def shuffle_data(x, y):
    shuf_idx = range(len(x))
    random.shuffle(shuf_idx)
    shuf_x = numpy.zeros(x.shape)
    shuf_y = numpy.zeros(y.shape)
    for i, idx in enumerate(shuf_idx):
        shuf_x[i,:,:,:] = x[idx,:,:,:]
        shuf_y[i,:] = y[idx,:]
    return shuf_x, shuf_y