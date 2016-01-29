import numpy as np
from skimage.io import imread
from joblib import Parallel, delayed
import lmdb
from utils import get_file_list
import caffe


def _write_batch_lmdb(db, batch):
    """
    Write a batch to an LMDB database
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for i, temp in enumerate(batch):
                datum, _id = temp
                key = str(_id)
                lmdb_txn.put(key, datum.SerializeToString())

    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        print('Doubling LMDB map size to %sMB ...' % (new_limit >> 20,))
        try:
            db.set_mapsize(new_limit)  # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0, 87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' %
                            lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_lmdb(db, batch)


def load_im_tuple(fname, key):
    im = imread(fname)
    image_datum = caffe.proto.caffe_pb2.Datum()
    image_datum.height = im.shape[0]
    image_datum.width = im.shape[1]
    image_datum.channels = im.shape[2]
    image_datum.data = im.tostring()
    return (image_datum, key)


def load_y_tuple(y, key):
    label_datum = caffe.io.array_to_datum(
        np.asarray(y, dtype=float).reshape(1, 1, 4))
    label_datum.height = 1
    label_datum.width = 4
    label_datum.channels = 1
    # label_datum.data = np.array(y).tostring()
    return (label_datum, key)


def process_batch(image_db, label_db, fnames_b, y_b):
    print "Reading the images and labels"
    with Parallel(n_jobs=-1) as parallel:
        Xb = parallel(delayed(load_im_tuple)
                      (fname, i) for i, fname in fnames_b)
        yb = parallel(delayed(load_y_tuple)(y, i) for i, y in y_b)
    print "Writing image data"
    _write_batch_lmdb(image_db, Xb)
    print "Writing label data"
    _write_batch_lmdb(label_db, yb)


def main(root_folder, batch_size=256, train_split=0.2):
    fnames, bboxes = get_file_list(root_folder)
    fnames = np.asarray(list(fnames))
    bboxes = np.asarray(list(bboxes), dtype=np.float32)
    num_samples = fnames.shape[0]
    num_val = int(round(num_samples * train_split))
    # Perform train validation split
    idx = np.arange(num_samples)
    rng = np.random.RandomState(seed=12345)
    rng.shuffle(idx)
    train_fnames = fnames[idx[num_val:]]
    train_bboxes = bboxes[idx[num_val:]]
    val_fnames = fnames[idx[:num_val]]
    val_bboxes = bboxes[idx[:num_val]]
    print "%d training samples and %d validation samples" % (train_fnames.shape[0], val_fnames.shape[0])
    # Create (key, value) pairs for storing in db
    X_t = []
    y_t = []
    for i in xrange(len(train_fnames)):
        X_t.append(('%08d' % i, train_fnames[i]))
        y_t.append(('%08d' % i, train_bboxes[i]))
    X_v = []
    y_v = []
    for i in xrange(len(val_fnames)):
        X_v.append(('%08d' % i, val_fnames[i]))
        y_v.append(('%08d' % i, val_bboxes[i]))

    # Training set
    train_image_db = lmdb.open('train_image', map_size=1e+12)
    train_label_db = lmdb.open('train_label', map_size=1e+12)

    prev_j = 0
    for j in xrange(batch_size, len(X_t), batch_size):
        print "Starting train batch #%d processing" % (prev_j / batch_size)
        process_batch(train_image_db, train_label_db,
                      X_t[prev_j:j], y_t[prev_j:j])
        prev_j = j

    train_image_db.close()
    train_label_db.close()
    # Validation set
    val_image_db = lmdb.open('val_image', map_size=1e+12)
    val_label_db = lmdb.open('val_label', map_size=1e+12)

    prev_j = 0
    for j in xrange(batch_size, len(X_v), batch_size):
        print "Starting val batch #%d processing" % (prev_j / batch_size)
        process_batch(val_image_db, val_label_db, X_v[prev_j:j], y_v[prev_j:j])
        prev_j = j

    val_image_db.close()
    val_label_db.close()

if __name__ == '__main__':
    root_folder = './train'
    main(root_folder, 1024, 0.2)
