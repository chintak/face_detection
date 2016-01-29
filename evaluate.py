import numpy as np
from skimage.io import imread, imsave
import lasagne
from nolearn.lasagne import NeuralNet
import cPickle as pickle
from models import nnet_three_conv_layer


def plot_learning_curves(nnet):
    pass


def load_network(fname):
    nnet = nnet_three_conv_layer()
    net_pkl = pickle.load(open(fname, 'rb'))
    nnet.load_params_from(net_pkl)
    return nnet
