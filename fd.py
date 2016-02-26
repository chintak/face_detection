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
from models import *
from helper import save_model_params, plot_weight_matrix_grid, plot_learning_curve

mumbai = timezone('Asia/Kolkata')
m_time = datetime.now(mumbai)


def train(X, y, config, max_epochs, batch_iterator='BatchIterator',
          pretrained_model=None, name=None, debug=True):
    # print globals()['net_name']
    global net_name
    sample = 500 if debug else X.shape[0]
    X_t, y_t = X[:sample], y[:sample, :]
    param_dump_folder = './model_%s' % (m_time.strftime(
        "%m_%d_%H_%M_%S") if name is None else name)

    print 'Model name: %s' % param_dump_folder
    print 'Debug mode:', debug
    # Load the net and add a function to save the params after every epoch
    nnet = globals()[config](batch_iterator, max_epochs)
    func_save_model = lambda n, h: save_model_params(
        n, h, param_dump_folder, debug)
    nnet.on_epoch_finished.append(func_save_model)
    func_learning_curve = lambda n, h: plot_learning_curve(
        n, h, param_dump_folder, debug)
    nnet.on_epoch_finished.append(func_learning_curve)
    # func_viz_weights = lambda n, h: plot_weight_matrix_grid(
    #     n, h, param_dump_folder, debug)
    # nnet.on_epoch_finished.append(func_viz_weights)

    print 'Config: %s' % config
    print 'Max num epochs: %d' % nnet.max_epochs
    print "Dataset loaded, shape:", X_t.shape, y_t.shape
    print "Loading pretrained model %s ..." % pretrained_model
    if pretrained_model is not None:
        pretrained_weights = pickle.load(open(pretrained_model, 'rb'))
        nnet.load_params_from(pretrained_weights)
    print "Finished loading"
    # Train
    if not debug:
        if not os.path.exists(param_dump_folder):
            os.mkdir(param_dump_folder)
    try:
        nnet.fit(X_t, y_t)
    except KeyboardInterrupt:
        pass
    if not debug:
        nnet.save_params_to(os.path.join(
            param_dump_folder, 'model_final.pkl'))
        # with open(os.path.join(param_dump_folder, 'full_nnet.pkl'), 'wb') as f:
        #     pickle.dump(nnet, f, -1)


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
    parser.add_argument('--config', type=str, choices=NET_CONFIGS,
                        default='nnet_4c3d_1233_convs_layer')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()
    if args.no_augment:
        train_preprocessed_img_lazy_batch(args.train_folder, max_epochs=args.epochs,
                                          name=args.name, config=args.config, debug=args.debug)
    else:
        train_original_imgs_with_augmentation(args.train_csv, args.test_csv, args.epochs, args.config,
                                              pretrained_model=args.pretrained_model,
                                              name=args.name, debug=args.debug)

# Notes
# python fd.py train/ -e 100 -n vgg_like_2
# - first good model
#   - biased with respect to location
