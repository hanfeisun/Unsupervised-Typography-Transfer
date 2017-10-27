from model.HAN import model_fn
from model.dataset import input_fn
from model.io import dump_image
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

model_params = {"learning_rate": 0.0002}

nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="./model_dir/HAN")
# save checkpoint per 10 minute
nn.train(input_fn=input_fn("./model_dir/train.obj"), steps=5000)
# nn.evaluate(input_fn=input_fn("./model_dir/val.obj"), steps=1)
#
transfers = nn.predict(input_fn=input_fn("./model_dir/val.obj", shuffle=False, num_epochs=1))

for i, t in enumerate(transfers):
    dump_image('./model_dir/%s.png' % i, t["g"])
