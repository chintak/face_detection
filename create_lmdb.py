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
    image_datum.data = im.tobytes()
    return (image_datum, key)


def load_y_tuple(y, key):
    label_datum = caffe.proto.caffe_pb2.Datum()
    label_datum.height = 1
    label_datum.width = 4
    label_datum.channels = 1
    label_datum.data = y.tobytes()
    return (label_datum, key)


def process_batch(image_db, label_db, fnames_b, y_b):
    print "Reading the images and labels"
    with Parallel(n_jobs=-1) as parallel:
        Xb = parallel(delayed(load_im_dict)
                      (fname, i) for i, fname in enumerate(fnames_b))
        yb = parallel(delayed(load_y_tuple)(y, i) for i, y in enumerate(y_b))
    print "Writing image data"
    _write_batch_lmdb(image_db, Xb)
    print "Writing label data"
    _write_batch_lmdb(label_db, yb)


def main(root_folder, batch_size=256):
    fnames, bboxes = get_file_list(train_folder)
    X = list(fnames)
    y = list(bboxes)
    num_samples = len(fnames)
    image_db = lmdb.open('train_image', map_size=1e+12)
    label_db = lmdb.open('train_label', map_size=1e+12)

    prev_j = 0
    for j in xrange(batch_size, num_samples, batch_size):
        print "Starting batch #%d processing" % (prev_j / batch_size)
        process_batch(image_db, label_db, X[prev_j:j], y[prev_j:j])
        prev_j = j

    image_db.close()
    label_db.close()

if __name__ == '__main__':
    root_foler = './train'
    main(root_folder)
