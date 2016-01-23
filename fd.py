import os
import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from pandas import read_csv
from skimage.io import imread
from joblib import delayed, Parallel

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


def _extract_names_bboxes(bname):
    df = read_csv(bname, sep=' ', names=['Name', 'BBox'])
    # exist_names = []
    df['Name'] = map(lambda n: os.path.join(
        os.path.dirname(bname), n), df['Name'])
    # for n in names:
    #     if os.path.exists(n):
    #         exist_names.append(n)
    # df['Name'] = exist_names
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
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('dense2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 3, 256, 256),  # 96x96 input pixels per batch
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dense1_num_units=500, dense2_num_units=500,
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=4,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        batch_iterator_train=BatchIterator(1),
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=1000,  # we want to train this many epochs
        verbose=1,
    )
    return net1


def load_im(name):
    im = imread(name)
    im = np.transpose(im, axes=[2, 0, 1])
    return im


def train(train_folder, config='nnet_three_conv_layer'):
    nnet = globals()[config]()
    fnames, bboxes = get_file_list(train_folder)
    y = np.array(list(bboxes))
    X = Parallel(n_jobs=-1)(delayed(load_im)(name)
                            for name in fnames[:10] if os.path.exists(name))
    X = np.array(X, dtype=np.uint8)
    print "Dataset loaded, shape:", X.shape, y.shape
    nnet.fit(X[:10, :, :, :], y[:10, :])

if __name__ == '__main__':
    train_folder = '/home/arya_03/data/faceScrub/train/'
    train(train_folder)
