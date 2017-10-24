from model import model_fn
from model.dataset import input_fn

import tensorflow as tf

model_params = {"learning_rate": 0.01}
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
nn.train(input_fn=input_fn("./model_dir/train.obj"), steps=100)
