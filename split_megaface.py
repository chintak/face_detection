import os
import glob
import pandas as pd
import json
import argparse
from joblib import Parallel, delayed
from pandas import read_csv
import numpy as np


def perform_split(csv_file, train_split_ratio):
    df = read_csv(csv_file, sep='\t')
    train_split = int(round(train_split_ratio * df.shape[0]))
    trainval_split = int(
        round((train_split_ratio + 0.5 * (1. - train_split_ratio)) * df.shape[0]))
    rng = np.random.RandomState(seed=12345)
    idx = np.arange(0, df.shape[0])
    rng.shuffle(idx)
    df = df.ix[idx]
    df = df.reset_index()
    df.ix[0:train_split].to_csv(
        '{}_train.csv'.format(csv_file.split('.')[0]), sep='\t', index=False)
    df.ix[train_split + 1:trainval_split].to_csv(
        '{}_val.csv'.format(csv_file.split('.')[0]), sep='\t', index=False)
    df.ix[trainval_split + 1:].to_csv(
        '{}_test.csv'.format(csv_file.split('.')[0]), sep='\t', index=False)


def read_megaface_jsons(js, i, tot):
    with open(js, 'r') as f:
        g = json.load(f)
    g['bounding_box']['x2'] = g['bounding_box'][
        'x'] + g['bounding_box']['width']
    g['bounding_box']['y2'] = g['bounding_box'][
        'y'] + g['bounding_box']['height']
    bbox = '{x:.1f},{y:.1f},{x2:.1f},{y2:.1f}'.format(**g['bounding_box'])
    name = js.replace('.jpg.json', '.jpg')
    if i % 100000 == 0:
        print "Read %d/%d" % (i, tot)
    return (name, bbox)


def all_dump(args):
    jsons = glob.glob(os.path.join(args.train_folder, '*/*/*.jpg.json'))
    tot = len(jsons)
    with open(args.list_csv, 'w', 0) as f:
        f.write('name\tbbox\n')
        for i, js in enumerate(jsons):
            name, bbox = read_megaface_jsons(js, i, tot)
            f.write('{}\t{}\n'.format(name, bbox))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder', help='Train folder root path')
    parser.add_argument('list_csv', help='Entire list of csv')
    parser.add_argument('-s', '--train_split',
                        help='Train/Val split ratio', type=float)
    args = parser.parse_args()
    args.train_split = 0.8 if args.train_split is None else args.train_split
    if not os.path.exists(args.list_csv):
        all_dump(args)
    else:
        print "%s File exists... Continue" % (args.list_csv)
    perform_split(args.list_csv, args.train_split)
