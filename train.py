import argparse
import tensorflow as tf
from model.HAN import model_fn
from model.dataset import input_fn
from model.io import dump_image
from subprocess import call

parser = argparse.ArgumentParser(description='Train HAN using pickled objects')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--steps', dest='steps', type=int, default=2000,
                    help='max training steps')
args = parser.parse_args()

model_params = {"learning_rate": args.lr}
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="./model_dir/HAN")

tf.logging.set_verbosity(tf.logging.INFO)
nn.train(input_fn=input_fn("./model_dir/train.obj"), max_steps=args.steps)

transfers = nn.predict(input_fn=input_fn("./model_dir/val.obj", shuffle=False, num_epochs=1))
for i, t in enumerate(transfers):
    dump_image('./model_dir/%s.png' % i, t["g"])

