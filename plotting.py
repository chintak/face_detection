import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread


def plot_face_bb(p, bb, path=True):
    im = p
    if path:
        im = cv2.imread(p)
    h, w, _ = im.shape
    cv2.rectangle(im, (int(bb[0] * h), int(bb[1] * w)),
                  (int(bb[2] * h), int(bb[3] * w)),
                  (255, 255, 0), thickness=4)
    plt.imshow(im[:, :, ::-1])
