import os
import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from pandas import read_csv
from skimage.io import imread
from joblib import delayed, Parallel
from utils import load_im
from lazy_batch_iterator import LazyBatchIterator

import lasagne
from lasagne import layers
from lasagne import init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator


def _extract_names_bboxes(bname):
    df = read_csv(bname, sep=' ', names=['Name', 'BBox'])
    df = df.dropna()
    df['Name'] = map(lambda n: os.path.join(
        os.path.dirname(bname), n), df['Name'])
    df['BBox'] = map(lambda ks: [np.float32(k)
                                 for k in ks.split(',')], df['BBox'])
    return df


def get_file_list(folder):
    names = os.listdir(folder)
    fnames = []
    bboxes = []
    bbox_names = map(lambda name: os.path.join(
        folder, name, '_bboxes.txt'), names)

    with Parallel(n_jobs=-1) as parallel:
        dfs = parallel(delayed(_extract_names_bboxes)(bname)
                       for bname in bbox_names if os.path.exists(bname))
    df = pd.concat(dfs, ignore_index=True)
    return df['Name'].values, df['BBox'].values


def nnet_three_conv_layer():
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            # ('conv5', layers.Conv2DLayer),
            # ('pool5', layers.MaxPool2DLayer),
            # ('conv6', Conv2DLayer),
            # ('pool6', MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('dense2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),  # 96x96 input pixels per batch
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        conv4_num_filters=128, conv4_filter_size=(2, 2), pool4_pool_size=(2, 2),
        # conv5_num_filters=128, conv5_filter_size=(2, 2), pool5_pool_size=(2, 2),
        # conv6_num_filters=128, conv6_filter_size=(2, 2), pool6_pool_size=(2,
        # 2),
        dense1_num_units=128, dense2_num_units=256,
        # output layer uses identity function
        output_nonlinearity=None,
        output_num_units=4,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        batch_iterator_train=LazyBatchIterator(batch_size=32),
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=30,  # we want to train this many epochs
        verbose=1,
    )
    return net1


def train(train_folder, config='nnet_three_conv_layer'):
    nnet = globals()[config]()
    fnames, bboxes = get_file_list(train_folder)
    y = np.array(list(bboxes), dtype=np.float32)
    assert (np.any(np.isnan(y)) or np.any(np.isinf(y))
            ) == False, "Invalid `y` detected"
    X = np.array(list(fnames))
    print "Dataset loaded, shape:", len(X), y.shape
    nnet.fit(X, y)

    # Training for 1000 epochs will take a while.  We'll pickle the
    # trained model so that we can load it back later:
    import cPickle as pickle
    with open('nnet.pickle', 'wb') as f:
        pickle.dump(nnet, f, -1)

if __name__ == '__main__':
    train_folder = 'train/'
    train(train_folder)
