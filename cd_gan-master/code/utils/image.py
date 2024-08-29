import itertools
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


def get_matrix(res_list):
    n = len(res_list)
    res_matrix = np.zeros(shape=(n, 4, 4))
    for i in range(n):
        a = res_list[i]
        res_matrix[i, :, :] += a
    return res_matrix


def show_and_save_image(res_mat, image_name, labels, show_bar):
    fig, ax = plt.subplots()
    im = ax.imshow(res_mat, interpolation='nearest', cmap="Blues")
    # cax = ax.matshow(res_mt)
    #     plt.title('Confusion matrix of the classifier')
    if show_bar:
        ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           xlabel='True label',
           ylabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = res_mat.max() / 1.5
    for i, j in itertools.product(range(len(labels)), range(len(labels))):
        plt.text(j, i, "{:0.1f}%".format(res_mat[i, j] * 100),
                 horizontalalignment="center",
                 color="white" if res_mat[i, j] > thresh else "black")
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # plt.colorbar()
    # plt.show()

    fig.savefig(image_name, bbox_inches='tight')
