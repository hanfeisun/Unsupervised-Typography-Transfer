# Unsupervised version of zi2zi

# Download the pretrained model from https://drive.google.com/file/d/0Bz6mX0EGe2ZuNEFSNWpTQkxPM2c/view
# then unzip it and put it under the project folder

import tensorflow as tf
sess = tf.Session()
saver = tf.train.import_meta_graph('./font27/gen_model-0.meta')
saver.restore(sess, tf.train.latest_checkpoint('./font27/'))
