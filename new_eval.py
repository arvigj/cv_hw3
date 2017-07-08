from __future__ import print_function

import numpy as np
import cv2
import subprocess
import itertools
from multiprocessing import Pool




import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


f = subprocess.check_output(["ls"]).split()
files = []
#make list of files that contain ellipse data
for i in f:
    if "ellipseList.txt" in i:
        files.append(i)
print(files)

class Image:
	def __init__(self, filename, window_size):
		self.im = cv2.imread(filename,0)
		#self.im = cv2.resize(self.im,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		self.mask = []
		self.mask_small = []
		self.windows = []
		self.windows_small = []
		self.scores = []
		self.scores_small = []
		self.cx = []
		self.cy = []
		self.decimation_factor = []
		self.imno = 0
		#self.slide = [-6,-4,-2,0,2,4,6]
		self.slide = [-3,-2,-1,0,1,2,3]
		self.window_size = window_size

	def ellipse(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0]),float(ellipse_info[1])]
		decim_fac = int(max(max(axes[0]*2/self.window_size,axes[1]*2/self.window_size),1))
		self.decimation_factor.append(decim_fac)

		#print "best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32)
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3]))
		self.cy.append(float(ellipse_info[4]))
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)
		#self.mask.append(mask[::2,::2])
		#self.cx[-1] /= 2
		#self.cy[-1] /= 2

	def ellipse_decim(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0])/2,float(ellipse_info[1])/2]
		print("best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32))
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3])/2)
		self.cy.append(float(ellipse_info[4])/2)
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)


	def get_score(self,mask,cx,cy,x,i,ellipse_size):
		s = self.window_size/2
		flag = False
		flag = flag or cy+x[0]-s < 0
		flag = flag or cx+x[0]-s < 0
		flag = flag or cy+x[1]+s+1 > mask.shape[0]
		flag = flag or cx+x[1]+s+1 > mask.shape[1]
		if flag == True:
			return -1.
		#intersect = np.sum(self.mask[i][cy+x[0]-16:cy+x[0]+17,cx+x[1]-16:cx+x[1]+17]).astype(float)
		#union = ellipse_size - intersect + (32*32)

		intersect = np.sum(mask[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1]).astype(float)
		union = ellipse_size - intersect + (4*s*s)
		self.imno += 1

		#CHOOSE THE SCORE YOU WANT
		return np.float32(intersect/union)
		#return intersect/ellipse_size

	def get_random_window(self,image,mask,center):
		s = self.window_size/2
		rand_mask = mask[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1]
		if rand_mask.size < (self.window_size**2) or np.sum(rand_mask) > 5:
			return None
		return image[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1].astype(np.float32)

	def get_windows(self):
		s = self.window_size/2
		self.image_slides = []
		self.score_slides = []
		for i in xrange(len(self.mask)):
			image = cv2.resize(self.im,(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA)
			mask = cv2.resize(self.mask[i].astype(np.uint8),(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA).astype(bool)
			mask_size = np.sum(mask)
			cx = int(round(self.cx[i]/self.decimation_factor[i]))
			cy = int(round(self.cy[i]/self.decimation_factor[i]))
			self.score_slides.append(map(lambda x: self.get_score(mask,cx,cy,x,i,mask_size), itertools.product(self.slide,self.slide)))
			self.image_slides.append(map(lambda x: image[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1].astype(np.float32), itertools.product(self.slide,self.slide)))
		
		#generate random images
		self.random_slides = []
		self.random_scores = []
		mask = np.zeros(self.im.shape)
		for i in xrange(len(self.mask)):
			mask = np.maximum(mask, self.mask[i].astype(int))
		mask = mask.astype(bool)
		rand = np.random.rand(self.imno,2)
		rand[:,0] *= self.im.shape[0]
		rand[:,1] *= self.im.shape[1]
		rand = rand.astype(int)
		iterate = 0
		goal = 2*self.imno
		while(self.imno < goal):
			try:
				randy = rand[iterate,0]
				randx = rand[iterate,1]
			except IndexError:
				rand = np.random.rand(self.imno,2)
				rand[:,0] *= self.im.shape[0]
				rand[:,1] *= self.im.shape[1]
				rand = rand.astype(int)
				iterate=0
				continue
			try:
				small = mask[randy-s:randy+s+1,randx-s:randx+s+1]
				#print "shape is %d %d"%(small.shape[0],small.shape[1])
				#print "val is %d"%np.sum(small)
			except IndexError:
				iterate+=1
				continue
			iterate+=1
			if small.size - (self.window_size**2) < 10:
				continue
			elif np.sum(small) > 10:
				continue
			self.random_slides.append(self.im[randy-s:randy+s+1,randx-s:randx+s+1].astype(np.float32))
			self.random_scores.append(np.float32(0))
			self.imno += 1
			#print "Adding random image"
			#print "%d left to go"%(goal-self.imno)

	def get_data(self):
		flatten = lambda l: [item for sublist in l for item in sublist]
		return flatten(self.image_slides)+self.random_slides, flatten(self.score_slides)+self.random_scores


def info(filename):
	with open(filename,"r") as f:
		slides = []
		scores = []
		while(True):
			try:
				imgpath = f.readline().split("\n")[0]+".jpg"
				if imgpath == ".jpg":
					return np.array(slides), np.array(scores)
				#print imgpath
				e = Image(imgpath,32)
				numfaces = f.readline().strip()
				#print numfaces
				print(numfaces)
				for i in xrange(int(numfaces)):
					ellipse_info = f.readline().split("\n")[0]
					#print ellipse_info
					e.ellipse(ellipse_info)
					#plt.imshow(e.im,cmap="gray",alpha=0.5)
					#plt.imshow(e.ellipse(ellipse_info),alpha=0.1,cmap="gray")
					#plt.show()
				e.get_windows()
				ims, im_scores = e.get_data()
				for i in xrange(len(ims)):
					slides.append(ims[i])
					scores.append(im_scores[i])
				#print
				#e.get_windows()
			except ValueError as a:
				#pass
				#    print e
				return
	#return

#info(files[0])
#exit()

pool = Pool(4)
a = np.array(pool.map(info,files[:2]))



images = np.concatenate(a[:,0]).tolist()
scores = np.concatenate(a[:,1]).tolist()

i=0
while(True):
	if i==len(images):
		break
	elif images[i].shape != (33,33):
		del images[i]
		del scores[i]
	else:
		i+=1

images = np.array(images)
scores = np.array(scores)




# images_flat = []
# scores_flat = []

# for i in xrange(len(images)):
# 	assert len(images[i]) == len(scores[i])
# 	for j in xrange(len(images[i])):
# 		print type(scores[i][j])
# 		images_flat.append(images[i][j])
# 		scores_flat.append(scores[i][j])

# images = np.array(images_flat)
# scores = np.array(scores_flat)

images = images[np.where(scores >= 0)]
scores = scores[np.where(scores >= 0)]
#scores_second = np.add(-1,scores)
#scores = np.concatenate((scores[:,np.newaxis],scores_second[:,np.newaxis]),axis=1)

#data = np.stack((images,scores[:,np.newaxis]),axis=1)
#np.random.shuffle(data)
#print(data.shape)


# plt.hist(scores,bins=50)
# plt.show()

# rand_range = (np.random.rand(10)*1000).astype(int)
# for i in xrange(10):
# 	print images[rand_range[i]].shape
# 	plt.imshow(images[rand_range[i]],cmap="gray",interpolation="nearest")
# 	print scores[rand_range[i]]
# 	plt.show()

print(scores.shape)
print(np.amin(scores))


def build_cnn(input_var=None):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	# Input layer, as usual:
	network = lasagne.layers.InputLayer(shape=(None, 1, 33, 33),
	                                    input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv2DLayer(
		network, num_filters=32, filter_size=(5, 5),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			network,
			num_units=1,
			nonlinearity=lasagne.nonlinearities.sigmoid)

	return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(data,model='cnn', num_epochs=500):
    # Load the dataset
    print("Loading data...")

    X = data[0].reshape(-1, 1, 33, 33)
    X /= np.float32(255)
    Y = np.round_(data[1]).astype(np.float32)
    #X = X.astype(np.float32)
    #Y = Y.astype(np.float32)
    # X_train = X[0:300000]
    # y_train = Y[0:300000]
    # X_val = X[-20000:]
    # y_val = Y[-20000:]
    # X_test = X[300000:400000]
    # y_test = Y[300000:400000]

    X_train = X[0:50000]
    y_train = Y[0:50000]
    X_val = X[-4000:]
    y_val = Y[-4000:]
    X_test = X[50000:80000]
    y_test = Y[50000:80000]


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fvector('targets')

    # Create neural network model (depending on first command line parameter)
    network = build_cnn(input_var)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_hinge_loss(prediction, target_var, log_odds=False)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_hinge_loss(test_prediction,
                                                            target_var, log_odds=False)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(test_prediction, target_var),
                      dtype=theano.config.floatX)
    #test_acc = T.mean(lasagne.objectives.binary_hinge_loss(prediction, target_var, log_odds=False), 
    #					dtype=theano.config.floatX)


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)



main([images,scores])

