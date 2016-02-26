import numpy as np
import lasagne
from nolearn.lasagne import NeuralNet
import cPickle as pickle
from models import *


def load_network(fname, config="nnet_4c3d_1233_convs_layer", batch_iterator="BatchIterator"):
    nnet = globals()[config](batch_iterator)
    net_pkl = pickle.load(open(fname, 'rb'))
    nnet.load_params_from(net_pkl)
    return nnet


def save_model_params(net, history, folder, debug=True):
    if not debug:
        net.save_params_to(os.path.join(
            folder, 'model_%d.pkl' % len(history)))


def plot_weight_matrix(Z, outname, save=True):
    num = Z.shape[0]
    fig = plt.figure(1, (80, 80))
    fig.subplots_adjust(left=0.05, right=0.95)
    grid = AxesGrid(fig, (1, 4, 2),  # similar to subplot(142)
                    nrows_ncols=(int(np.ceil(num / 10.)), 10),
                    axes_pad=0.04,
                    share_all=True,
                    label_mode="L",
                    )

    for i in range(num):
        im = grid[i].imshow(Z[i, :, :, :].mean(
            axis=0), cmap='gray')
    for i in range(grid.ngrids):
        grid[i].axis('off')

    for cax in grid.cbar_axes:
        cax.toggle_label(False)
    if save:
        fig.savefig(outname, bbox_inches='tight')
        fig.clear()


def plot_weight_matrix_grid(net, history, folder, debug=True):
    """
    A grid of 2x2 images with a single colorbar
    """
    if debug:
        return
    params = net.get_all_params_values()
    convs = [k for k in params.keys() if 'conv' in k]
    outdir = os.path.join(folder, 'outputs', 'epoch_%d' % (len(history)))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with Parallel(n_jobs=3) as parallel:
        parallel(delayed(plot_weight_matrix)(params[k][0],
                                             os.path.join(outdir, 'weights_%s.png' % k))
                 for k in convs)


def plot_learning_curve(_, history, folder, debug=True):
    arr = np.asarray(
        map(lambda k: [k['epoch'], k['train_loss'], k['valid_loss']], history))
    plt.figure()
    plt.plot(arr[:, 0], arr[:, 1], 'r', marker='o',
             label='Training loss', linewidth=2.0)
    plt.plot(arr[:, 0], arr[:, 2], 'b', marker='o',
             label='Validation loss', linewidth=2.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0.0, np.max(arr[:, 1:]) * 1.3])
    plt.title('Learning curve')
    plt.legend()
    if not debug:
        plt.savefig('%s/learning_curve.png' % folder, bbox_inches='tight')
        plt.close()
