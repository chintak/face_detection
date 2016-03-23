from __future__ import division
import os
import sys
import cv2
import caffe
import numpy as np
import theano
from skimage.io import imread
from nolearn.lasagne import BatchIterator
from joblib import Parallel, delayed
from utils import create_fixed_image_shape

# Load the VGG_FACE caffe net
# model_file = "data/VGG_FACE_deploy.prototxt"
# weight_file = "data/VGG_FACE.caffemodel"
# if (not os.path.exists(model_file)) or (not os.path.exists(weight_file)):
#     raise AttributeError("VGG_FACE prototxt (%s) and the caffemodel (%s) files are required" % (
#         model_file, weight_file))
# # caffe.set_mode_cpu()
# if 'cpu' in theano.config.device:
#     print "Caffe -- CPU mode"
#     caffe.set_mode_cpu()
# else:
#     device = int(theano.config.device.split('gpu')[-1])
#     print "Caffe -- GPU mode, device %d" % device
#     caffe.set_mode_gpu()
#     caffe.set_device(device)
model_file = "../data/VGG_FACE_deploy.prototxt"
weight_file = "../data/VGG_FACE.caffemodel"
caffe.set_mode_cpu()
vgg_net = caffe.Net(model_file, weight_file, caffe.TEST)


def extract_vgg_face_features(imgs):
    global vgg_net
    vgg_net.blobs['data'].reshape(*imgs.shape)
    vgg_net.blobs['data'].data[...] = imgs
    out = vgg_net.forward()
    return vgg_net.blobs['pool4'].data.copy()


class VggFeatBatchIterator(BatchIterator):
    """docstring for VggFeatBatchIterator"""

    def __init__(self, batch_size):
        super(VggFeatBatchIterator, self).__init__(batch_size)
        self.img_size = (224, 224, 3)
        self.MAX_FACE_SIZE = 200
        self.MIN_FACE_SIZE = 64
        self.rng_seeder = np.random.RandomState(seed=1234)

    def compute_scale_factor(self, img, face_width, hc, wc, high_scale=None, low_scale=None):
        # Possible scales of computation
        high_scale = min(high_scale or self.MAX_FACE_SIZE, self.MAX_FACE_SIZE)
        high_scale = high_scale / 2. / face_width
        low_scale = max(low_scale or self.MIN_FACE_SIZE, self.MIN_FACE_SIZE)
        low_scale = low_scale / 2. / face_width
        assert high_scale > low_scale, "Same scale detected %.3f, %.3f" % (
            low_scale, high_scale)
        scale_comp = self.rng.choice(
            np.arange(low_scale, high_scale, (high_scale - low_scale) / 100), 1)[0]

        new_face_width = round(face_width * scale_comp)
        swc, shc = round(wc * scale_comp), round(hc * scale_comp)
        res = cv2.resize(img, None, fx=scale_comp, fy=scale_comp)
        return res, new_face_width, shc, swc

    def compute_translation(self, face_width, shc, swc, min_pad=None):
        # Possible location of the face
        h0, w0, _ = self.img_size
        min_pad = min_pad or face_width + 5
        lw, lh, hw, hh = (min(min_pad, w0 - min_pad), min(min_pad, h0 - min_pad),
                          max(min_pad, w0 - min_pad), max(min_pad, h0 - min_pad))
        twc, thc = self.rng.randint(lw, hw, 2)
        return thc, twc

    def copy_source_to_target(self, res, new_face_width, shc, swc, thc, twc):
        # Compute the top left and bottom right coordinates for source and
        # target imgs
        h, w, _ = res.shape
        h0, w0, _ = self.img_size
        sfh = shc - thc
        tfh = int(0 if sfh > 0 else abs(sfh))
        sfh = int(0 if sfh < 0 else sfh)
        sfw = swc - twc
        tfw = int(0 if sfw > 0 else abs(sfw))
        sfw = int(0 if sfw < 0 else sfw)
        seh = shc - thc + h0
        teh = int(h0 if seh <= h else h0 - seh + h)
        seh = int(h if seh > h else seh)
        sew = swc - twc + w0
        tew = int(w0 if sew <= w else w0 - sew + w)
        sew = int(w if sew > w else sew)
        new_bb = np.array([twc - new_face_width, thc - new_face_width,
                           twc + new_face_width, thc + new_face_width], dtype=np.float32)
        out = np.random.randint(
            0, high=255, size=self.img_size).astype(np.uint8)
        out[tfh:teh, tfw:tew, :] = res[sfh:seh, sfw:sew, :]
        return out, new_bb

    def get_scaled_translated_img_bb(self, name, bb):
        im = imread(name)
        img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        h0, w0, _ = self.img_size
        wc, hc = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        face_width = (bb[3] - bb[1]) / 2
        # Old approach: scale and then translate
        res, new_face_width, shc, swc = self.compute_scale_factor(
            img, face_width, hc, wc)
        thc, twc = self.compute_translation(new_face_width, shc, swc)
        # New approach: translate and then scale
        # thc, twc = self.compute_translation(face_width, hc, wc,
        #                                     min_pad=self.MIN_FACE_SIZE + 10)
        # high_scale = np.min([thc - 5, h0 - thc - 5, twc - 5, w0 - twc - 5])
        # res, new_face_width, shc, swc = self.compute_scale_factor(
        #     img, face_width, hc, wc,
        #     high_scale=high_scale, low_scale=None)
        out_bgr, new_bb = self.copy_source_to_target(res, new_face_width,
                                                     shc, swc, thc, twc)

        log = "%.1f,%.1f,%.0f\n" % (
            (new_bb[1] + new_bb[3]) / 2, (new_bb[0] + new_bb[2]) / 2, new_face_width * 2)
        with open('aug.csv', mode='a', buffering=0) as f:
            f.write(log)
        # cv2.rectangle(out_bgr, (int(new_bb[0]), int(new_bb[1])), (int(new_bb[2]), int(new_bb[3])),
        #               (255, 255, 0), thickness=4)
        # cv2.imwrite("%d.jpg" % os.getpid(), out_bgr)
        # sys.exit(0)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return out, new_bb

    def transform(self, Xb, yb):
        X_n, y_n = super(VggFeatBatchIterator, self).transform(Xb, yb)
        if y_n is not None:
            # Train mode
            ret = Parallel(n_jobs=-1)(delayed(load_augment_im)(self, name, bb)
                                      for name, bb in zip(X_n, y_n))
            Xb_ims = np.asarray(map(lambda v: v[0], ret))
            Xb_ims = np.transpose(Xb_ims, [0, 3, 1, 2])
            Xb = extract_vgg_face_features(Xb_ims)
            yb = np.asarray(map(lambda v: v[1], ret))
        else:
            # Test mode
            ret = Parallel(n_jobs=-1)(delayed(prepare_test_img)(self, name)
                                      for name in X_n)
            Xb_ims = np.asarray(ret)
            Xb_ims = np.transpose(Xb_ims, [0, 3, 1, 2])
            Xb = extract_vgg_face_features(Xb_ims)
            yb = None
        return Xb, yb


def load_augment_im(self, name, bb=None):
    self.rng_seeder.seed()
    seed = self.rng_seeder.randint(0, 10000)
    self.rng = np.random.RandomState(seed=seed)
    x, new_bb = self.get_scaled_translated_img_bb(name, bb)
    y = new_bb / self.img_size[0]  # Range: 0 to 1
    # y = (new_bb / self.img_size[0] - 0.5) / 0.5  # Range: -1 to 1

    return (x, y)


def prepare_test_img(self, name):
    im = imread(name)
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    res = create_fixed_image_shape(
        img, frame_size=self.img_size, random_fill=True, mode='fit')
    # cv2.imwrite("%d.jpg" % os.getpid(), res)
    # sys.exit(0)
    return res
