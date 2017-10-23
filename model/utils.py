import scipy.misc as misc
import numpy as np


def normalize_image(img):
    """
    Make image zero centered and in between (0, 1)
    """
    normalized = img / 255.
    return normalized


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B

