import os
import pandas as pd
from pandas import read_csv
import glob
from joblib import Parallel, delayed
import numpy as np
import argparse


def read_facescrub_img_list(folder, actor_label_txt, actress_label_txt, accept_pattern='*.jpg'):
    full_names = glob.glob(os.path.join(folder, accept_pattern))
    only_names = map(lambda f: os.path.splitext(
        os.path.basename(f))[0], full_names)

    pd_male = read_csv(actor_label_txt, sep='\t')
    del pd_male['url'], pd_male['image_id'], pd_male['face_id']
    pd_female = read_csv(actress_label_txt, sep='\t')
    del pd_female['url'], pd_female['image_id'], pd_female['face_id']
    pd_celeb = pd.concat([pd_male, pd_female], ignore_index=True)
    pd_celeb = pd_celeb.drop_duplicates(
        subset='sha256', keep='last').set_index('sha256')

    bboxes = map(lambda k: pd_celeb.bbox[k], only_names)
    return full_names, bboxes


def perform_split(args):
    fnames, bboxes = read_facescrub_img_list(
        args.train_folder, args.actor_label_path, args.actress_label_path, accept_pattern='*/*.jpg')
    np_fnames = np.asarray(fnames)
    np_bboxes = np.asarray(bboxes)
    train_split = int(round(args.train_split * len(fnames)))
    rng = np.random.RandomState(seed=1234)
    idx = np.arange(0, len(fnames))
    rng.shuffle(idx)
    X = np_fnames[idx]
    y = np_bboxes[idx]
    df = pd.DataFrame({'name': X, 'bbox': y})
    df.ix[:train_split].to_csv('train.csv', sep='\t', index=False)
    df.ix[train_split + 1:].to_csv('val.csv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder', help='Train folder root path')
    parser.add_argument('actor_label_path', help='Path to actors label list')
    parser.add_argument('actress_label_path',
                        help='Path to actresses label list')
    parser.add_argument('-s', '--train_split',
                        help='Train/Val split ratio', type=float)
    args = parser.parse_args()
    args.train_split = 0.8 if args.train_split is None else args.train_split
    perform_split(args)
