"""
cifar-10 dataset related code goes here
"""

#uncomment this only when running download_cifar. Obstructs TF functionality.
#from keras.datasets import cifar10
import pickle
from os.path import expanduser
"""
download cifar10 data from keras and shape result
"""
def download_cifar():
	keras = 'theano'
	reshape_height = 32
	reshape_width = 32
	reshape_channels = 3
	
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	if keras == 'theano':
		X_train = X_train.transpose(0,2,3,1)
		X_test = X_test.transpose(0,2,3,1)
		
	y_train_vector = numpy.zeros((len(y_train),10))
	y_test_vector = numpy.zeros((len(y_test),10))
	
	for i in range(len(y_train)):
		y_train_vector[i,y_train[i]] = 1
	for i in range(len(y_test)):
		y_test_vector[i,y_test[i]] = 1
		
	X_train_new = numpy.zeros((len(y_train),reshape_height,reshape_width,reshape_channels))
	X_test_new = numpy.zeros((len(y_test),reshape_height,reshape_width,reshape_channels))
		
	for i in range(len(X_train)):
		if i % 100 == 0:
			print ("upsampling training image " + str(i))
		X_train_new[i,:,:,:] = scipy.misc.imresize(X_train[i,:,:,:], [reshape_height,reshape_width,reshape_channels])
	
	for i in range(len(X_test)):
		if i % 100 == 0:
			print ("upsampling testing image " + str(i))
		X_test_new[i,:,:,:] = scipy.misc.imresize(X_test[i,:,:,:], [reshape_height,reshape_width,reshape_channels])
	
	x = [(X_train_new, y_train_vector), (X_test_new,y_test_vector)]
	
	pickle.dump(x, open(expanduser("~/adversary/data/cifar_data.pkl"),"w"))

"""
load cifar10 data from desktop.
"""
def load_cifar():
	x = pickle.load(open(expanduser("~/adversary/data/cifar_data.pkl"),"r"))
	return x
