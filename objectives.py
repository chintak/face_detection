import numpy as np
import theano
import theano.tensor as T


def iou_loss(p, t):
    # print "pass"
    tp, tt = p.reshape((p.shape[0], 2, 2)), t.reshape((t.shape[0], 2, 2))
    overlaps_t0 = T.maximum(tp[:, 0, :], tt[:, 0, :])
    overlaps_t1 = T.minimum(tp[:, 1, :], tt[:, 1, :])
    intersection = overlaps_t1 - overlaps_t0
    bool_overlap = T.min(intersection, axis=1) > 0
    intersection = intersection[:, 0] * intersection[:, 1]
    intersection = T.maximum(intersection, np.float32(0.))
    dims_p = tp[:, 1, :] - tp[:, 0, :]
    areas_p = dims_p[:, 0] * dims_p[:, 1]
    dims_t = tt[:, 1, :] - tt[:, 0, :]
    areas_t = dims_t[:, 0] * dims_t[:, 1]
    union = areas_p + areas_t - intersection
    loss = 1. - T.minimum(
        T.exp(T.log(T.abs_(intersection)) -
              T.log(T.abs_(union) + np.float32(1e-5))),
        np.float32(1.)
    )
    # return loss
    return T.mean(loss)


def iou_loss_val(p, t):
    tp, tt = p.reshape((p.shape[0], 2, 2)), t.reshape((t.shape[0], 2, 2))
    overlaps = np.zeros_like(tp, dtype=np.float32)
    overlaps[:, 0, :] = np.maximum(tp[:, 0, :], tt[:, 0, :])
    overlaps[:, 1, :] = np.minimum(tp[:, 1, :], tt[:, 1, :])
    intersection = overlaps[:, 1, :] - overlaps[:, 0, :]
    bool_overlap = np.min(intersection, axis=1) > 0
    intersection = intersection[:, 0] * intersection[:, 1]
    intersection = np.maximum(intersection, 0.)
    # print "bool", bool_overlap
    # print "Int", intersection
    dims_p = tp[:, 1, :] - tp[:, 0, :]
    areas_p = dims_p[:, 0] * dims_p[:, 1]
    dims_t = tt[:, 1, :] - tt[:, 0, :]
    areas_t = dims_t[:, 0] * dims_t[:, 1]
    union = areas_p + areas_t - intersection
    # print "un", union
    loss = 1. - np.minimum(
        np.exp(np.log(np.abs(intersection)) - np.log(np.abs(union) + 1e-5)),
        1.
    )
    # print loss
    return np.mean(loss)


def smooth_l1_loss(predictions, targets, sigma=1.5):
    cond = np.float32(1. / sigma / sigma)
    point_five = np.float32(0.5)
    sigma_t = np.float32(sigma)
    sub_const = np.float32(0.5 / sigma / sigma)
    diff = T.abs_(predictions - targets)
    out = T.switch(T.lt(diff, cond),
                   point_five * sigma_t * diff * sigma_t * diff,
                   diff - sub_const)
    return T.mean(T.sum(out, axis=1))


def smooth_l1_loss_val(predictions, targets, sigma=1.5):
    assert predictions.shape == targets.shape, (
        "Shape mismatch: predicted values %s and target values %s" % (
            predictions.shape, targets.shape))
    diff = np.abs(predictions - targets)
    out = diff.copy()
    cond = 1. / sigma / sigma
    out[diff < cond] = (0.5 * sigma * diff * sigma * diff)[diff < cond]
    out[diff >= cond] = diff[diff >= cond] - 0.5 / sigma / sigma
    return np.mean(out.sum(axis=1))
