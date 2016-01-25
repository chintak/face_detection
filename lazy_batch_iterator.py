from nolearn.lasagne import BatchIterator
import numpy as np
from utils import load_im
from joblib import Parallel, delayed
import os


class LazyBatchIterator(BatchIterator):
    """docstring for LazyBatchIterator"""

    def transform(self, Xb, yb):
        X_n, yb = super(LazyBatchIterator, self).transform(Xb, yb)
        Xb = Parallel(n_jobs=-1)(delayed(load_im)(name)
                                 for name in X_n)
        Xb = np.array(Xb, dtype=np.float32) / 256.
        assert (np.any(np.isnan(Xb)) or np.any(np.isinf(Xb))
                ) == False, "Invalid `Xb` detected"
        return Xb, yb
