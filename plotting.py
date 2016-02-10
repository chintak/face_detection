import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_face_bb(p, bb, scale=True, path=True):
    if path:
        im = cv2.imread(p)
    else:
        im = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
    if scale:
        h, w, _ = im.shape
        cv2.rectangle(im, (int(bb[0] * h), int(bb[1] * w)),
                      (int(bb[2] * h), int(bb[3] * w)),
                      (255, 255, 0), thickness=4)
        # print bb * np.asarray([h, w, h, w])
    else:
        cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      (255, 255, 0), thickness=4)
        print "no"
    plt.figure()
    plt.imshow(im[:, :, ::-1])
