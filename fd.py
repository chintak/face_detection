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
from models import nnet_4c3d_1233_convs_layer


mumbai = timezone('Asia/Kolkata')
m_time = datetime.now(mumbai)


def train(X_t, y_t, config, max_epochs, batch_iterator='BatchIterator', name=None, debug=True):
    print 'Model name: %s' % name
    print 'Debug mode:', debug
    nnet = globals()[config](batch_iterator, max_epochs)
    print 'Config: %s' % config
    print 'Max num epochs: %d' % nnet.max_epochs
    print "Dataset loaded, shape:", X_t.shape, y_t.shape
    # Train loop, save params every 5 epochs
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


def train_original_imgs_with_augmentation(train_folder, max_epochs=30, name=None, config='nnet_4c3d_1233_convs_layer', debug=True):
    pass


def train_preprocessed_img_lazy_batch(train_folder, max_epochs=30, name=None, config='nnet_4c3d_1233_convs_layer', debug=True):
    fnames, bboxes = get_file_list(train_folder)
    y = np.array(list(bboxes), dtype=np.float32)
    assert (np.any(np.isnan(y)) or np.any(np.isinf(y))
            ) == False, "Invalid `y` detected"
    X = np.array(list(fnames))
    sample = 500 if debug else X.shape[0]
    X_t, y_t = X[:sample], y[:sample, :]
    train(X_t, y_t, config, max_epochs, 'LazyBatchIterator', name, debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder', type=str)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    if not args.augment:
        train_preprocessed_img_lazy_batch(args.train_folder, max_epochs=args.epochs,
                                          name=args.name, debug=args.debug)
    else:
        train_original_imgs_with_augmentation(args.train_folder, max_epochs=args.epochs,
                                              name=args.name, debug=args.debug)

# Notes
# python fd.py train/ -e 100 -n vgg_like_2
# - first good model
#   - biased with respect to location
