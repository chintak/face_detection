import matplotlib as mpl
mpl.use('Agg')
import os
import numpy as np
import pandas as pd
from pandas import read_csv
from skimage.io import imread
from joblib import delayed, Parallel
import argparse
from pytz import timezone
from datetime import datetime
import cPickle as pickle

from utils import get_file_list
from models import nnet_4c3d_1233_convs_layer, nnet_4c3d_1234_convs_layer

mumbai = timezone('Asia/Kolkata')
m_time = datetime.now(mumbai)


def train(X, y, config, max_epochs, batch_iterator='BatchIterator',
          pretrained_model=None, name=None, debug=True):
    sample = 500 if debug else X.shape[0]
    X_t, y_t = X[:sample], y[:sample, :]
    param_dump_folder = './model_%s' % (m_time.strftime(
        "%m_%d_%H_%M_%S") if name is None else name)
    print 'Model name: %s' % param_dump_folder
    print 'Debug mode:', debug
    nnet = globals()[config](batch_iterator, max_epochs)
    if pretrained_model is not None:
        pretrained_weights = pickle.load(open(pretrained_model, 'rb'))
        nnet.load_params_from(pretrained_weights)
    print 'Config: %s' % config
    print 'Max num epochs: %d' % nnet.max_epochs
    print "Dataset loaded, shape:", X_t.shape, y_t.shape
    # Train loop, save params every 5 epochs
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
        nnet.save_params_to(os.path.join(
            param_dump_folder, 'model_%d.pkl' % len(nnet.train_history_)))
        with open(os.path.join(param_dump_folder, 'full_nnet.pkl'), 'wb') as f:
            pickle.dump(nnet, f, -1)


def train_original_imgs_with_augmentation(train_csv, test_csv, max_epochs, config,
                                          pretrained_model=None, name=None, debug=True):
    train_df = read_csv(train_csv, sep='\t')
    X = np.asarray(train_df['name'].as_matrix())
    y_str = train_df['bbox']
    y_l = map(lambda k: [np.float32(v)
                         for v in k.split(',')], y_str)
    y = np.asarray(y_l)
    with open('aug.csv', mode='w', buffering=0) as f:
        f.write("r,c,width\n")
    train(X, y, config, max_epochs, 'AugmentBatchIterator',
          pretrained_model, name, debug)


def train_preprocessed_img_lazy_batch(train_folder, max_epochs, config, name=None, debug=True):
    fnames, bboxes = get_file_list(train_folder)
    y = np.array(list(bboxes), dtype=np.float32)
    assert (np.any(np.isnan(y)) or np.any(np.isinf(y))
            ) == False, "Invalid `y` detected"
    X = np.array(list(fnames))
    train(X, y, config, max_epochs, 'LazyBatchIterator', name, debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--config', type=str,
                        default='nnet_4c3d_1233_convs_layer')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--pretrained-model-config', type=str,
                        default='nnet_4c3d_1233_convs_layer')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    if not args.augment:
        train_preprocessed_img_lazy_batch(args.train_folder, max_epochs=args.epochs,
                                          name=args.name, config=args.config, debug=args.debug)
    else:
        train_original_imgs_with_augmentation(args.train_csv, args.test_csv, args.epochs, args.config,
                                              pretrained_model=args.pretrained_model, name=args.name, debug=args.debug)

# Notes
# python fd.py train/ -e 100 -n vgg_like_2
# - first good model
#   - biased with respect to location
