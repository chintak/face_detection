import os
import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from pandas import read_csv
from joblib import delayed, Parallel
# from lasagne import layers
# from nolearn.lasagne import BatchIterator
# from nolearn.lasagne import NeuralNet

# try:
#     from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
#     from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
# except ImportError:
#     Conv2DLayer = layers.Conv2DLayer
#     MaxPool2DLayer = layers.MaxPool2DLayer


def _extract_names_bboxes(bname):
    df = read_csv(bname, sep=' ', names=['Name', 'BBox'])
    df['Name'] = map(lambda n: os.path.join(
        os.path.dirname(bname), n), df['Name'])
    df['BBox'] = map(lambda ks: [float(k)
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
    return df['Name'], df['BBox']
