import pickle
import numpy as np
import tensorflow as tf
import random
import os
from enum import Enum
from tensorflow.contrib.data import Dataset


class PairPolicy(Enum):
    STRONG_PAIR = 1
    SOFT_PAIR_MINUS = 2
    SOFT_PAIR_PLUS = 3
    NO_PAIR = 4


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


generator_x = None
generator_batch_size = None
generator_num_epochs = None

def generator():
    # Total number of training examples
    size = generator_x['source'].shape[0]
    # Run forever
    while True:
        idx = np.random.randint(size, size=generator_batch_size)
        x_mb = {}
        x_mb['source'] = generator_x['source'][idx,:,:,:]
        x_mb['target'] = generator_x['target'][idx,:,:,:].copy()
        np.random.shuffle(x_mb['target'])
        yield x_mb


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


def input_fn(path, shuffle=True, num_epochs=None, pair_policy=PairPolicy.STRONG_PAIR, batch_size=8):
    pickled = PickledImageProvider(path)
    processed = [np.array(process(e[1])).astype(np.float32) for e in pickled.examples]
    x = {
        'source': np.array([img[0] for img in processed]),
        'target': np.array([img[1] for img in processed])
    }

    if pair_policy is PairPolicy.STRONG_PAIR:
        return tf.estimator.inputs.numpy_input_fn(
            x=x,
            y=None,
            shuffle=shuffle,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
    elif pair_policy is PairPolicy.SOFT_PAIR_MINUS:
        global generator_x
        generator_x = x
        global generator_batch_size
        generator_batch_size = batch_size
        global generator_num_epochs
        generator_num_epochs = num_epochs

        shape = (batch_size,) + x['source'].shape[1:]
        x_dataset = Dataset.from_generator(
            generator=generator, 
            output_types={'source': tf.float32, 'target': tf.float32},
            output_shapes={'source': shape, 'target': shape}
        )
        return x_dataset.make_one_shot_iterator
    elif pair_policy is PairPolicy.SOFT_PAIR_PLUS:
        raise NotImplemented


