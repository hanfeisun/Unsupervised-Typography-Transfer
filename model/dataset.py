import pickle
import numpy as np
import tensorflow as tf
import random
import os
from .utils import bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def process(img):
    img = bytes_to_file(img)
    try:
        img_A, img_B = read_split_image(img)
        shape = img_A.shape
        img_A = np.array(normalize_image(img_A)).reshape((shape[0], shape[1], -1))
        img_B = np.array(normalize_image(img_B)).reshape((shape[0], shape[1], -1))
        return img_A, img_B
    finally:
        img.close()


def input_fn(path):
    pickled = PickledImageProvider(path)
    labels = np.array([e[0] for e in pickled.examples])
    processed = [np.array(process(e[1])).astype(np.float32) for e in pickled.examples]
    x = {
        'source': np.array([img[0] for img in processed]),
        'target': np.array([img[1] for img in processed])
    }

    return tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=labels,
        batch_size=2,
        shuffle=False,
        num_epochs=1
    )
