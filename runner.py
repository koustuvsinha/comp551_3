##
## Implementation of Lenet architecture
## Modified from Theano Tutorial https://www.kaggle.com/c/digit-recognizer/forums/t/10552/convolutional-neural-networks-using-theano/130455
#

import theano.tensor as T
from theano import function
from theano import shared
import theano
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import os
import time
from pandas import DataFrame, Series
import pandas as pd
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from lenet import LeNetConvPoolLayer
from hidden import HiddenLayer
from logistic import LogisticRegression

class LenetArch:
    def __init__(self):
        self.learning_rate=0.1
        directory='###FIXME###'
        self.nkerns=[20, 50]
        self.batch_size=500
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                                gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        rng = numpy.random.RandomState(23455)

        datasets = self.load_data()

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                                # [int] labels

        ishape = (60, 60)  # this is the size of MNIST images

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = x.reshape((batch_size, 1, 60, 60))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (60-5+1,60-5+1)=(56,56)
        # maxpooling reduces this further to (56/2,56/2) = (28,28)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                    image_shape=(batch_size, 1, 60, 60),
                    filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        ## special layer for us
        #layer01 = LeNetConvPoolLayer(rng, input=layer0.output,
        #            image_shape=(batch_size, 1, 28, 28),
        #            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                    image_shape=(batch_size, nkerns[0], 28, 28),
                    filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 12 * 12,
                                 n_out=500, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=19)

        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        validate_model = theano.function([index], layer3.errors(y),
                    givens={
                        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                        y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

        # create a list of all model parameters to be fit by gradient descent
        params = layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        train_model = theano.function([index], cost, updates=updates,
                  givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    def load_data(self):
        ''' Loads the dataset
        '''

        #############
        # LOAD DATA #
        #############

        print '... loading data'

        # Load the dataset
        x_train = numpy.fromfile('train_x.bin',dtype='uint8')
        x_train = x_train.reshape((100000,60,60)).reshape((100000,-1))
        train_df = DataFrame(x_train)
        x_test = numpy.fromfile('test_x.bin',dtype='uint8')
        x_test = x_test.reshape((20000,60,60)).reshape((20000,-1))
        test_df = DataFrame(x_test)
        train_y = DataFrame.from_csv('train_y.csv',index_col=False)
        train_set = [train_df.values[0:70000, 0:] / 255., train_y.values[0:70000, 1]]
        valid_set = [train_df.values[70000:, 0:] / 255., train_y.values[70000:, 1]]
        test_set = test_df.values / 255.
        #train_set, valid_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x = theano.shared(numpy.asarray(test_set,
                                                 dtype=theano.config.floatX),
                                   borrow=True)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                test_set_x]
        return rval

    def train(self):
        n_epochs=200
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                   # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                          # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                            in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, validation error %f %%' % \
                              (epoch, this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break
