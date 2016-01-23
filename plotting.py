import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_face_bb(p, bb, path=True):
    im = p
    if path:
        im = cv2.imread(p)
    if isinstance(bb[0], float):
        h, w, _ = im.shape
        cv2.rectangle(im, (int(bb[0] * h), int(bb[1] * w)),
                      (int(bb[2] * h), int(bb[3] * w)),
                      (255, 255, 0), thickness=4)
    else:
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                      (255, 255, 0), thickness=4)
    plt.imshow(im[:, :, ::-1])
