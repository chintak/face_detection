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

from vgg_feat_batch_iterator import VggFeatBatchIterator
from helper import AdjustVariable, StepVariableUpdate
from sklearn.metrics import mean_squared_error
from objectives import iou_loss_val, smooth_l1_loss, smooth_l1_loss_val

from joblib import Parallel, delayed


NET_CONFIGS = [
    'config_2c_2_2d_smoothl1_lr_step',
]


def config_2c_2_2d_smoothl1_lr_step(batch_iterator="BatchIterator", max_epochs=30):
    custom_batch_iterator = globals()[batch_iterator]
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1_1', layers.Conv2DLayer),
            ('conv1_2', layers.Conv2DLayer),
            ('conv1_3', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            # ('conv2_1', layers.Conv2DLayer),
            # ('conv2_2', layers.Conv2DLayer),
            # ('pool2', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            # ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 512, 14, 14),
        conv1_1_num_filters=512, conv1_1_filter_size=(3, 3), conv1_1_pad=(1, 1),
        conv1_2_num_filters=512, conv1_2_filter_size=(3, 3), conv1_2_pad=(1, 1),
        conv1_3_num_filters=512, conv1_3_filter_size=(3, 3), conv1_3_pad=(1, 1),
        pool1_pool_size=(2, 2), pool1_stride=(2, 2),
        # conv2_1_num_filters=128, conv2_1_filter_size=(3, 3), conv2_1_pad=(1, 1),
        # conv2_2_num_filters=128, conv2_2_filter_size=(3, 3), conv2_2_pad=(1, 1),
        # pool2_pool_size=(3, 3), pool2_stride=(2, 2),
        # conv1_1_W=Orthogonal(gain=1.0),
        # conv1_2_W=Orthogonal(gain=1.0),
        # conv1_2_W=Orthogonal(gain=1.0),
        # conv2_1_W=Orthogonal(gain=1.0),
        # conv2_2_W=Orthogonal(gain=1.0),

        dense1_num_units=2048,  # drop1_p=0.5,
        dense2_num_units=2048,  # drop2_p=0.5,
        dense2_nonlinearity=lasagne.nonlinearities.sigmoid,
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,  # 4 bbox coordinates and 1 for confidence

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.001)),
        update_momentum=theano.shared(np.float32(0.9)),

        batch_iterator_train=custom_batch_iterator(batch_size=32),
        batch_iterator_test=custom_batch_iterator(batch_size=16),
        on_epoch_finished=[
            # StepVariableUpdate('update_learning_rate', changes={
            #     10: 0.0001,
            #     40: 0.0001
            # }),
            StepVariableUpdate('update_momentum', changes={
                10: 0.95,
                40: 0.975
            })
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
