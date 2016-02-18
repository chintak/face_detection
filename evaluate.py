import numpy as np
import lasagne
from nolearn.lasagne import NeuralNet
import cPickle as pickle
from models import *


def load_network(fname, config="nnet_4c3d_1233_convs_layer", batch_iterator="BatchIterator"):
    nnet = globals()[config](batch_iterator)
    net_pkl = pickle.load(open(fname, 'rb'))
    nnet.load_params_from(net_pkl)
    return nnet
