import numpy as np
import cv2
from skimage.io import imread
import os
from joblib import Parallel, delayed
import pandas as pd
from pandas import read_csv


def _extract_names_bboxes(bname):
    df = read_csv(bname, sep=' ', names=['Name', 'BBox'])
    df['Name'] = map(lambda n: os.path.join(
        os.path.dirname(bname), n), df['Name'])
    df['BBox'] = map(lambda ks: [(np.float32(k) - 0.5) / 0.5
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
                       for bname in bbox_names)
    df = pd.concat(dfs, ignore_index=True)
    df['Flag'] = df['Name'].map(lambda x: True if os.path.exists(x) else False)
    print "Initial number of images:", df['Name'].count()
    df = df[df['Flag'] == True]
    print "Total number of existing images:", df['Name'].count()
    return df['Name'].values, df['BBox'].values


def load_im_chw(name):
    im = imread(name)
    im = np.transpose(im, axes=[2, 0, 1])
    return im


def create_fixed_image_shape(img, frame_size=(200, 200, 3), random_fill=True, mode='crop'):
    if mode == 'fit':
        X1, Y1, _ = frame_size
        if random_fill:
            image_frame = np.random.randint(0, high=255, size=frame_size)
        else:
            image_frame = np.zeros(frame_size, dtype='uint8')

        X2, Y2 = img.shape[1], img.shape[0]

        if X2 > Y2:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))
        else:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))

        img = cv2.resize(img, (X_new, Y_new))
        img = img.reshape((img.shape[0], img.shape[1], frame_size[2]))

        X_space_center = ((X1 - X_new) / 2)
        Y_space_center = ((Y1 - Y_new) / 2)

        # print Y_new, X_new, Y_space_center, X_space_center
        image_frame[Y_space_center: Y_space_center + Y_new,
                    X_space_center: X_space_center + X_new] = img

    elif mode == 'crop':
        X1, Y1, _ = frame_size
        image_frame = np.zeros(frame_size, dtype='uint8')

        X2, Y2 = img.shape[1], img.shape[0]

        # increase the size of smaller length (width or hegiht)
        if X2 > Y2:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))
        else:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))

        img = cv2.resize(img, (X_new, Y_new))
        img = img.reshape((img.shape[0], img.shape[1], frame_size[2]))

        X_space_clip = (X_new - X1) / 2
        Y_space_clip = (Y_new - Y1) / 2

        # trim image both top, down, left and right
        if X_space_clip == 0 and Y_space_clip != 0:
            img = img[Y_space_clip:-Y_space_clip, :, :]
        elif Y_space_clip == 0 and X_space_clip != 0:
            img = img[:, X_space_clip:-X_space_clip, :]

        if img.shape[0] != X1:
            img = img[1:, :]
        if img.shape[1] != Y1:
            img = img[:, 1:]

        image_frame[:, :] = img
    return image_frame
