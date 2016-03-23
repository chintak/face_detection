from __future__ import division
import os
import sys
import cv2
import caffe
import numpy as np
from pandas import read_csv
from skimage.io import imread
from joblib import Parallel, delayed
import json
from config import cfg


class AugmentDataLayer(caffe.Layer):
    """docstring for AugmentDataLayer"""

    def _compute_scale_factor(self, img, face_width, hc, wc, high_scale=None, low_scale=None):
        # Possible scales of computation
        high_scale = min(high_scale or self._MAX_FACE_SIZE,
                         self._MAX_FACE_SIZE)
        high_scale = high_scale / 2. / face_width
        low_scale = max(low_scale or self._MIN_FACE_SIZE, self._MIN_FACE_SIZE)
        low_scale = low_scale / 2. / face_width
        assert high_scale > low_scale, "Same scale detected %.3f, %.3f" % (
            low_scale, high_scale)
        scale_comp = self._rng.choice(
            np.arange(low_scale, high_scale, (high_scale - low_scale) / 100), 1)[0]

        new_face_width = round(face_width * scale_comp)
        swc, shc = round(wc * scale_comp), round(hc * scale_comp)
        res = cv2.resize(img, None, fx=scale_comp, fy=scale_comp)
        return res, new_face_width, shc, swc

    def _compute_translation(self, face_width, shc, swc, min_pad=None):
        # Possible location of the face
        h0, w0, _ = self._img_size
        min_pad = min_pad or face_width + 5
        lw, lh, hw, hh = (min(min_pad, w0 - min_pad), min(min_pad, h0 - min_pad),
                          max(min_pad, w0 - min_pad), max(min_pad, h0 - min_pad))
        twc, thc = self._rng.randint(lw, hw, 2)
        return thc, twc

    def _copy_source_to_target(self, res, new_face_width, shc, swc, thc, twc):
        # Compute the top left and bottom right coordinates for source and
        # target imgs
        h, w, _ = res.shape
        h0, w0, _ = self._img_size
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
            0, high=255, size=self._img_size).astype(np.uint8)
        out[tfh:teh, tfw:tew, :] = res[sfh:seh, sfw:sew, :]
        return out, new_bb

    def _get_scaled_translated_img_bb(self, name, bb):
        im = imread(name)
        img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        h0, w0, _ = self._img_size
        wc, hc = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        face_width = (bb[3] - bb[1]) / 2
        # Old approach: scale and then translate
        res, new_face_width, shc, swc = self._compute_scale_factor(
            img, face_width, hc, wc)
        thc, twc = self._compute_translation(new_face_width, shc, swc)
        # New approach: translate and then scale
        # thc, twc = self._compute_translation(face_width, hc, wc,
        #                                     min_pad=self._MIN_FACE_SIZE + 10)
        # high_scale = np.min([thc - 5, h0 - thc - 5, twc - 5, w0 - twc - 5])
        # res, new_face_width, shc, swc = self._compute_scale_factor(
        #     img, face_width, hc, wc,
        #     high_scale=high_scale, low_scale=None)
        out_bgr, new_bb = self._copy_source_to_target(res, new_face_width,
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

    def _shuffle_db_inds(self):
        self._perm = self._rng_seeder.permutation(
            np.arange(self._num_train_samples))
        self._cur = 0

    def _set_imagedb(self, source_train_file):
        df = read_csv(source_train_file, sep='\t')
        self._imdb_name = df.name.as_matrix()
        self._bbdb = np.asarray(
            map(lambda ks: [np.float32(k) for k in ks.split(',')],
                df.bbox), dtype=np.float32)
        self._num_train_samples = self._imdb_name.shape[0]
        assert self._imdb_name.shape[0] == self._bbdb.shape[0], (
            "Equal number of files and bboxes required")
        self._shuffle_db_inds()
        print "Total number of training samples {}".format(
            self._num_train_samples)

    def setup(self, bottom, top):
        self._rng_seeder = np.random.RandomState(seed=cfg.RNG_SEED)
        assert len(bottom) == 0, "No bottom required"
        assert len(top) == 2, "2 tops required"
        self._name_to_top_map = {}
        self._name_to_top_map['data'] = 0
        self._name_to_top_map['gt_boxes'] = 1

        layer_params = json.loads(self.param_str_) or {}
        assert layer_params.has_key('source'), (
            "List of training images expected. "
            "File format: /path/to/img\\tx1,y1,x2,y2"
        )
        source_train_file = layer_params['source']
        assert os.path.exists(source_train_file), (
            "%s file does not exist" % (source_train_file))
        self._set_imagedb(source_train_file)

        self._epochs = 0
        self._MAX_FACE_SIZE = cfg.TRAIN.MAX_FACE_SIZE
        self._MIN_FACE_SIZE = cfg.TRAIN.MIN_FACE_SIZE
        self._batch_size = cfg.TRAIN.BATCH_SIZE
        self._img_size = cfg.IMG_SIZE

        top[0].reshape(
            self._batch_size,
            self._img_size[2],
            self._img_size[0],
            self._img_size[1]
        )
        top[1].reshape(self._batch_size, 4)

    def _get_next_minibatch_inds(self):
        if self._cur + self._batch_size >= self._num_train_samples:
            self._shuffle_db_inds()
            self._epochs += 1
            print "Epochs:", self._epochs

        mb_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        return mb_inds

    def _get_next_minibatch(self):
        mb_inds = self._get_next_minibatch_inds()
        Xs = self._imdb_name[mb_inds]
        ys = self._bbdb[mb_inds]
        ret = Parallel(n_jobs=-1, backend='threading')(
            delayed(load_augment_im)(self, name, bb)
            for name, bb in zip(Xs, ys))
        Xb_ims = np.asarray(map(lambda v: v[0], ret), dtype=np.float32) / 255.
        blobs = {}
        blobs['data'] = np.transpose(Xb_ims, [0, 3, 1, 2])
        blobs['gt_boxes'] = np.asarray(map(lambda v: v[1], ret))
        return blobs

    def forward(self, bottom, top):
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].reshape(*(blob.shape))
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, bottom, propogate_down, top):
        pass

    def reshape(self, bottom, top):
        pass


def load_augment_im(self, name=None, bb=None):
    self._rng_seeder.seed()
    seed = self._rng_seeder.randint(0, 10000)
    self._rng = np.random.RandomState(seed=seed)
    x, new_bb = self._get_scaled_translated_img_bb(name, bb)
    y = new_bb / self._img_size[0]  # Range: 0 to 1
    # y = (new_bb / self.img_size[0] - 0.5) / 0.5  # Range: -1 to 1

    return (x, y)
