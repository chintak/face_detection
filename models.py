import numpy as np
import os
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.init import Orthogonal
from lasagne.updates import nesterov_momentum
from sklearn.metrics import mean_squared_error

from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from joblib import Parallel, delayed

from nolearn.lasagne import BatchIterator
from lazy_batch_iterator import LazyBatchIterator
from augment_batch_iterator import AugmentBatchIterator


def iou_loss(p, t):
    # print "pass"
    tp, tt = p.reshape((p.shape[0], 2, 2)), t.reshape((t.shape[0], 2, 2))
    overlaps_t0 = T.maximum(tp[:, 0, :], tt[:, 0, :])
    overlaps_t1 = T.minimum(tp[:, 1, :], tt[:, 1, :])
    intersection = overlaps_t1 - overlaps_t0
    bool_overlap = T.min(intersection, axis=1) > 0
    intersection = intersection[:, 0] * intersection[:, 1]
    intersection = T.maximum(intersection, np.float32(0.))
    dims_p = tp[:, 1, :] - tp[:, 0, :]
    areas_p = dims_p[:, 0] * dims_p[:, 1]
    dims_t = tt[:, 1, :] - tt[:, 0, :]
    areas_t = dims_t[:, 0] * dims_t[:, 1]
    union = areas_p + areas_t - intersection
    loss = 1. - T.minimum(
        T.exp(T.log(T.abs_(intersection)) -
              T.log(T.abs_(union) + np.float32(1e-5))),
        np.float32(1.)
    )
    # return loss
    return T.mean(loss)


def iou_loss_val(p, t):
    tp, tt = p.reshape((p.shape[0], 2, 2)), t.reshape((t.shape[0], 2, 2))
    overlaps = np.zeros_like(tp, dtype=np.float32)
    overlaps[:, 0, :] = np.maximum(tp[:, 0, :], tt[:, 0, :])
    overlaps[:, 1, :] = np.minimum(tp[:, 1, :], tt[:, 1, :])
    intersection = overlaps[:, 1, :] - overlaps[:, 0, :]
    bool_overlap = np.min(intersection, axis=1) > 0
    intersection = intersection[:, 0] * intersection[:, 1]
    intersection = np.maximum(intersection, 0.)
    # print "bool", bool_overlap
    # print "Int", intersection
    dims_p = tp[:, 1, :] - tp[:, 0, :]
    areas_p = dims_p[:, 0] * dims_p[:, 1]
    dims_t = tt[:, 1, :] - tt[:, 0, :]
    areas_t = dims_t[:, 0] * dims_t[:, 1]
    union = areas_p + areas_t - intersection
    # print "un", union
    loss = 1. - np.minimum(
        np.exp(np.log(np.abs(intersection)) - np.log(np.abs(union) + 1e-5)),
        1.
    )
    # print loss
    return np.mean(loss)


def smooth_l1_loss(predictions, targets, sigma=1.5):
    cond = np.float32(1. / sigma / sigma)
    point_five = np.float32(0.5)
    sigma_t = np.float32(sigma)
    sub_const = np.float32(0.5 / sigma / sigma)
    diff = T.abs_(predictions - targets)
    out = T.switch(T.lt(diff, cond),
                   point_five * sigma_t * diff * sigma_t * diff,
                   diff - sub_const)
    return T.mean(T.sum(out, axis=1))


def smooth_l1_loss_val(predictions, targets, sigma=1.5):
    assert predictions.shape == targets.shape, (
        "Shape mismatch: predicted values %s and target values %s" % (
            predictions.shape, targets.shape))
    diff = np.abs(predictions - targets)
    out = diff.copy()
    cond = 1. / sigma / sigma
    out[diff < cond] = (0.5 * sigma * diff * sigma * diff)[diff < cond]
    out[diff >= cond] = diff[diff >= cond] - 0.5 / sigma / sigma
    return np.mean(out.sum(axis=1))


def nnet_4c3d_1234_convs_layer(batch_iterator="BatchIterator", max_epochs=30):
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
        update_learning_rate=0.001,
        update_momentum=0.9,

        batch_iterator_train=custom_batch_iterator(batch_size=32),
        batch_iterator_test=custom_batch_iterator(batch_size=32),
        objective_loss_function=smooth_l1_loss,
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


def nnet_4c3d_1233_convs_layer(batch_iterator="BatchIterator", max_epochs=30):
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
        on_epoch_finished=[plot_learning_curve],
        regression=True,
        max_epochs=max_epochs,
        verbose=1,
    )
    return net1
