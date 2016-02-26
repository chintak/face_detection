import numpy as np
import os
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
from helper import AdjustVariable
from sklearn.metrics import mean_squared_error
from objectives import iou_loss_val, smooth_l1_loss, smooth_l1_loss_val

from joblib import Parallel, delayed


NET_CONFIGS = [
    'config_4c_1234_3d_smoothl1_lr_linear',
    'config_4c_1234_3d_squaredloss_lr_linear',
    'config_4c_1233_3d'
]


def config_4c_1234_3d_smoothl1_lr_linear(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
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
            # ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            # ('drop2', layers.DropoutLayer),
            ('dense3', layers.DenseLayer),
            # ('drop3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),
        conv1_1_num_filters=86, conv1_1_filter_size=(5, 5), conv1_1_stride=(2, 2),
        conv1_2_num_filters=104, conv1_2_filter_size=(3, 3), conv1_2_pad=(1, 1), pool1_pool_size=(2, 2), pool1_stride=(2, 2),
        conv2_1_num_filters=128, conv2_1_filter_size=(3, 3), conv2_1_pad=(1, 1),
        conv2_2_num_filters=128, conv2_2_filter_size=(3, 3), conv2_2_pad=(1, 1),
        pool2_pool_size=(3, 3), pool2_stride=(2, 2),
        conv3_1_num_filters=256, conv3_1_filter_size=(3, 3), conv3_1_pad=(1, 1),
        conv3_2_num_filters=256, conv3_2_filter_size=(3, 3), conv3_2_pad=(1, 1),
        conv3_3_num_filters=256, conv3_3_filter_size=(3, 3), conv3_3_pad=(1, 1),
        pool3_pool_size=(3, 3), pool3_stride=(2, 2),
        conv4_1_num_filters=196, conv4_1_filter_size=(3, 3), conv4_1_pad=(1, 1),
        conv4_2_num_filters=196, conv4_2_filter_size=(3, 3), conv4_2_pad=(1, 1),
        conv4_3_num_filters=196, conv4_3_filter_size=(3, 3), conv4_3_pad=(1, 1),
        conv4_4_num_filters=196, conv4_4_filter_size=(3, 3), conv4_4_pad=(1, 1),
        pool4_pool_size=(2, 2), pool4_stride=(2, 2),
        conv1_1_W=Orthogonal(gain=1.0),
        conv1_2_W=Orthogonal(gain=1.0),
        conv2_1_W=Orthogonal(gain=1.0),
        conv2_2_W=Orthogonal(gain=1.0),
        conv3_1_W=Orthogonal(gain=1.0),
        conv3_2_W=Orthogonal(gain=1.0),
        conv3_3_W=Orthogonal(gain=1.0),
        conv4_1_W=Orthogonal(gain=1.0),
        conv4_2_W=Orthogonal(gain=1.0),
        conv4_3_W=Orthogonal(gain=1.0),
        conv4_4_W=Orthogonal(gain=1.0),

        dense1_num_units=4096,  # drop1_p=0.5,
        dense2_num_units=2048,  # drop2_p=0.5,
        dense3_num_units=512,   # drop3_p=0.5,
        dense3_nonlinearity=lasagne.nonlinearities.sigmoid,
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.001)),
        update_momentum=theano.shared(np.float32(0.9)),

        batch_iterator_train=custom_batch_iterator(batch_size=72),
        batch_iterator_test=custom_batch_iterator(batch_size=48),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.001, stop=0.00005),
            AdjustVariable('update_momentum', start=0.9, stop=0.98)
        ],
        objective_loss_function=smooth_l1_loss,
        # objective_loss_function=iou_loss,
        custom_scores=[
            # ('smoothl1', smooth_l1_loss_val),
            ('iou_loss', iou_loss_val),
            ('squared_error', mean_squared_error)
        ],
        regression=True,
        max_epochs=max_epochs,
        verbose=1,
    )
    return net1


def config_4c_1234_3d_squaredloss_lr_linear(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
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
            # ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            # ('drop2', layers.DropoutLayer),
            ('dense3', layers.DenseLayer),
            # ('drop3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),
        conv1_1_num_filters=86, conv1_1_filter_size=(5, 5), conv1_1_stride=(2, 2),
        conv1_2_num_filters=104, conv1_2_filter_size=(3, 3), conv1_2_pad=(1, 1), pool1_pool_size=(2, 2), pool1_stride=(2, 2),
        conv2_1_num_filters=128, conv2_1_filter_size=(3, 3), conv2_1_pad=(1, 1),
        conv2_2_num_filters=128, conv2_2_filter_size=(3, 3), conv2_2_pad=(1, 1),
        pool2_pool_size=(3, 3), pool2_stride=(2, 2),
        conv3_1_num_filters=256, conv3_1_filter_size=(3, 3), conv3_1_pad=(1, 1),
        conv3_2_num_filters=256, conv3_2_filter_size=(3, 3), conv3_2_pad=(1, 1),
        conv3_3_num_filters=256, conv3_3_filter_size=(3, 3), conv3_3_pad=(1, 1),
        pool3_pool_size=(3, 3), pool3_stride=(2, 2),
        conv4_1_num_filters=196, conv4_1_filter_size=(3, 3), conv4_1_pad=(1, 1),
        conv4_2_num_filters=196, conv4_2_filter_size=(3, 3), conv4_2_pad=(1, 1),
        conv4_3_num_filters=196, conv4_3_filter_size=(3, 3), conv4_3_pad=(1, 1),
        conv4_4_num_filters=196, conv4_4_filter_size=(3, 3), conv4_4_pad=(1, 1),
        pool4_pool_size=(2, 2), pool4_stride=(2, 2),
        conv1_1_W=Orthogonal(gain=1.0),
        conv1_2_W=Orthogonal(gain=1.0),
        conv2_1_W=Orthogonal(gain=1.0),
        conv2_2_W=Orthogonal(gain=1.0),
        conv3_1_W=Orthogonal(gain=1.0),
        conv3_2_W=Orthogonal(gain=1.0),
        conv3_3_W=Orthogonal(gain=1.0),
        conv4_1_W=Orthogonal(gain=1.0),
        conv4_2_W=Orthogonal(gain=1.0),
        conv4_3_W=Orthogonal(gain=1.0),
        conv4_4_W=Orthogonal(gain=1.0),

        dense1_num_units=4096,  # drop1_p=0.5,
        dense2_num_units=2048,  # drop2_p=0.5,
        dense3_num_units=512,   # drop3_p=0.5,
        dense3_nonlinearity=lasagne.nonlinearities.sigmoid,
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.001)),
        update_momentum=theano.shared(np.float32(0.9)),

        batch_iterator_train=custom_batch_iterator(batch_size=72),
        batch_iterator_test=custom_batch_iterator(batch_size=48),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.001, stop=0.00005),
            AdjustVariable('update_momentum', start=0.9, stop=0.98)
        ],
        # objective_loss_function=smooth_l1_loss,
        # objective_loss_function=iou_loss,
        custom_scores=[
            ('smoothl1', smooth_l1_loss_val),
            ('iou_loss', iou_loss_val),
            ('squared_error', mean_squared_error)
        ],
        regression=True,
        max_epochs=max_epochs,
        verbose=1,
    )
    return net1


def config_4c_1233_3d(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[
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
        input_shape=(None, 3, 256, 256),
        conv1_num_filters=86, conv1_filter_size=(5, 5), conv1_stride=(2, 2), conv1_pad=(1, 1), pool1_pool_size=(2, 2),
        conv2_1_num_filters=128, conv2_1_filter_size=(3, 3), conv2_1_pad=(1, 1),
        conv2_2_num_filters=128, conv2_2_filter_size=(3, 3), conv2_2_pad=(1, 1),
        pool2_pool_size=(2, 2),
        conv3_1_num_filters=256, conv3_1_filter_size=(3, 3), conv3_1_pad=(1, 1),
        conv3_2_num_filters=256, conv3_2_filter_size=(3, 3), conv3_2_pad=(1, 1),
        conv3_3_num_filters=256, conv3_3_filter_size=(3, 3), conv3_3_pad=(1, 1),
        pool3_pool_size=(2, 2),
        conv4_1_num_filters=196, conv4_1_filter_size=(3, 3), conv4_1_pad=(1, 1),
        conv4_2_num_filters=196, conv4_2_filter_size=(3, 3), conv4_2_pad=(1, 1),
        conv4_3_num_filters=196, conv4_3_filter_size=(3, 3), conv4_3_pad=(1, 1),
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
        output_num_units=4,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,

        batch_iterator_train=custom_batch_iterator(batch_size=64),
        batch_iterator_test=custom_batch_iterator(batch_size=64),
        regression=True,
        max_epochs=max_epochs,
        verbose=1,
    )
    return net1
