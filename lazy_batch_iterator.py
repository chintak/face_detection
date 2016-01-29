from nolearn.lasagne import BatchIterator
import numpy as np
from skimage.io import imread
from joblib import Parallel, delayed
import os


def load_im_f(name):
    im = imread(name).astype(np.float32) / 256.
    im = np.transpose(im, [2, 0, 1])
    return im


class LazyBatchIterator(BatchIterator):
    """docstring for LazyBatchIterator"""

    def transform(self, Xb, yb):
        X_n, yb = super(LazyBatchIterator, self).transform(Xb, yb)
        Xb = Parallel(n_jobs=-1)(delayed(load_im_f)(name)
                                 for name in X_n)
        Xb = np.asarray(Xb)
        return Xb, yb
