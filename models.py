import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.init import Orthogonal
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from nolearn.lasagne import BatchIterator
from lazy_batch_iterator import LazyBatchIterator
from augment_batch_iterator import AugmentBatchIterator


def nnet_4c3d_1233_convs_layer(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2_1', layers.Conv2DLayer),
            ('conv2_2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3_1', layers.Conv2DLayer),
            ('conv3_2', layers.Conv2DLayer),
            ('conv3_3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4_1', layers.Conv2DLayer),
            ('conv4_2', layers.Conv2DLayer),
            ('conv4_3', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('dense2', layers.DenseLayer),
            ('dense3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),  # 96x96 input pixels per batch
        conv1_num_filters=86, conv1_filter_size=(5, 5), conv1_stride=(2, 2), pool1_pool_size=(2, 2),
        conv2_1_num_filters=128, conv2_1_filter_size=(2, 2),
        conv2_2_num_filters=128, conv2_2_filter_size=(2, 2),
        pool2_pool_size=(2, 2),
        conv3_1_num_filters=256, conv3_1_filter_size=(2, 2),
        conv3_2_num_filters=256, conv3_2_filter_size=(2, 2),
        conv3_3_num_filters=256, conv3_3_filter_size=(2, 2),
        pool3_pool_size=(2, 2),
        conv4_1_num_filters=196, conv4_1_filter_size=(2, 2),
        conv4_2_num_filters=196, conv4_2_filter_size=(2, 2),
        conv4_3_num_filters=196, conv4_3_filter_size=(2, 2),
        pool4_pool_size=(2, 2),
        conv1_W=Orthogonal(gain=1.0),
        conv2_1_W=Orthogonal(gain=1.0),
        conv2_2_W=Orthogonal(gain=1.0),
        conv3_1_W=Orthogonal(gain=1.0),
        conv3_2_W=Orthogonal(gain=1.0),
        conv3_3_W=Orthogonal(gain=1.0),
        conv4_1_W=Orthogonal(gain=1.0),
        conv4_2_W=Orthogonal(gain=1.0),
        conv4_3_W=Orthogonal(gain=1.0),
        dense1_num_units=2048, dense2_num_units=1024, dense3_num_units=512,
        # dense1_nonlinearity=lasagne.nonlinearities.rectify,
        # dense2_nonlinearity=lasagne.nonlinearities.rectify,
        dense3_nonlinearity=lasagne.nonlinearities.sigmoid,
        dense1_W=Orthogonal(gain=1.0),
        dense2_W=Orthogonal(gain=1.0),
        dense3_W=Orthogonal(gain=1.0),
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,

        batch_iterator_train=custom_batch_iterator(batch_size=64),
        batch_iterator_test=custom_batch_iterator(batch_size=64),
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=max_epochs,  # we want to train this many epochs
        verbose=1,
    )
    return net1


def nnet_4c3d_1234_convs_layer(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2_1', layers.Conv2DLayer),
            ('conv2_2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3_1', layers.Conv2DLayer),
            ('conv3_2', layers.Conv2DLayer),
            ('conv3_3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4_1', layers.Conv2DLayer),
            ('conv4_2', layers.Conv2DLayer),
            ('conv4_3', layers.Conv2DLayer),
            ('conv4_4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('dense2', layers.DenseLayer),
            ('dense3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),  # 96x96 input pixels per batch
        conv1_num_filters=86, conv1_filter_size=(5, 5), conv1_stride=(2, 2), pool1_pool_size=(2, 2),
        conv2_1_num_filters=128, conv2_1_filter_size=(2, 2),
        conv2_2_num_filters=128, conv2_2_filter_size=(2, 2),
        pool2_pool_size=(2, 2),
        conv3_1_num_filters=256, conv3_1_filter_size=(2, 2),
        conv3_2_num_filters=256, conv3_2_filter_size=(2, 2),
        conv3_3_num_filters=256, conv3_3_filter_size=(2, 2),
        pool3_pool_size=(2, 2),
        conv4_1_num_filters=196, conv4_1_filter_size=(2, 2),
        conv4_2_num_filters=196, conv4_2_filter_size=(2, 2),
        conv4_3_num_filters=196, conv4_3_filter_size=(2, 2),
        conv4_4_num_filters=196, conv4_4_filter_size=(2, 2),
        pool4_pool_size=(2, 2),
        conv1_W=Orthogonal(gain=1.0),
        conv2_1_W=Orthogonal(gain=1.0),
        conv2_2_W=Orthogonal(gain=1.0),
        conv3_1_W=Orthogonal(gain=1.0),
        conv3_2_W=Orthogonal(gain=1.0),
        conv3_3_W=Orthogonal(gain=1.0),
        conv4_1_W=Orthogonal(gain=1.0),
        conv4_2_W=Orthogonal(gain=1.0),
        conv4_3_W=Orthogonal(gain=1.0),
        conv4_4_W=Orthogonal(gain=1.0),
        dense1_num_units=2048, dense2_num_units=1024, dense3_num_units=512,
        # dense1_nonlinearity=lasagne.nonlinearities.rectify,
        # dense2_nonlinearity=lasagne.nonlinearities.rectify,
        dense3_nonlinearity=lasagne.nonlinearities.sigmoid,
        dense1_W=Orthogonal(gain=1.0),
        dense2_W=Orthogonal(gain=1.0),
        dense3_W=Orthogonal(gain=1.0),
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,

        batch_iterator_train=custom_batch_iterator(batch_size=64),
        batch_iterator_test=custom_batch_iterator(batch_size=64),
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=max_epochs,  # we want to train this many epochs
        verbose=1,
    )
    return net1
