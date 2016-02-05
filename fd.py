import os
import numpy as np
import pandas as pd
from pandas import read_csv
from skimage.io import imread
from joblib import delayed, Parallel
import argparse
from pytz import timezone
from datetime import datetime

from utils import get_file_list
from models import nnet_three_conv_layer


mumbai = timezone('Asia/Kolkata')
m_time = datetime.now(mumbai)


def train(train_folder, max_epochs=30, name=None, config='nnet_three_conv_layer', debug=True):
    print 'Model name: %s' % name
    print 'Config: %s' % config
    print 'Debug mode:', debug
    print 'Max num epochs: %d' % max_epochs
    nnet = globals()[config]()
    nnet.max_epochs = max_epochs
    fnames, bboxes = get_file_list(train_folder)
    y = np.array(list(bboxes), dtype=np.float32)
    assert (np.any(np.isnan(y)) or np.any(np.isinf(y))
            ) == False, "Invalid `y` detected"
    X = np.array(list(fnames))
    sample = 500 if debug else X.shape[0]
    X_t, y_t = X[:sample], y[:sample, :]
    print "Dataset loaded, shape:", X_t.shape, y_t.shape
    # nnet.fit(X_t, y_t)
    # Train loop, save params every 2 epochs
    param_dump_folder = './model_%s' % (m_time.strftime(
        "%m_%d_%H_%M_%S") if name is None else name)
    if not debug:
        if not os.path.exists(param_dump_folder):
            os.mkdir(param_dump_folder)
    for i in range(1, nnet.max_epochs, 5):
        try:
            nnet.fit(X_t, y_t, epochs=5)
            if not debug:
                nnet.save_params_to(os.path.join(
                    param_dump_folder, 'model_%d.pkl' % len(nnet.train_history_)))
        except KeyboardInterrupt:
            break
    # Training for 1000 epochs will take a while.  We'll pickle the
    # trained model so that we can load it back later:
    if not debug:
        import cPickle as pickle
        with open(os.path.join(param_dump_folder, 'full_nnet.pkl'), 'wb') as f:
            pickle.dump(nnet, f, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder', type=str)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    train(args.train_folder, max_epochs=args.epochs,
          name=args.name, debug=args.debug)
