# Unsupervised version of zi2zi

# Download the pretrained model from https://drive.google.com/file/d/0Bz6mX0EGe2ZuNEFSNWpTQkxPM2c/view
# then unzip it and put it under the project folder

import tensorflow as tf
import argparse

from zi2ziNet.unet import UNet

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--src-font', dest='source_front', required=True)
parser.add_argument('--text', dest='text', required=True)
parser.add_argument('--checkpoint', dest='checkpoint', required=True)


args = parser.parse_args()


def inspect_graph():
    from pprint import pprint
    var_list = tf.global_variables()
    collection_list = tf.get_default_graph().get_all_collection_keys()
    pprint(var_list)
    pprint(tf.get_default_graph().get_tensor_by_name("no_target_A_and_B_images:0"))


def main(_):
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        model = UNet()
        model.register_session(sess)
        model.build_model()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=model.retrieve_generator_vars())

        src_obj = None
        model.infer(source_obj="/home/ec2-user/10707/zi2ziu_experiment/data/val.obj", model_dir=args.checkpoint, embedding_ids=[0], save_dir="./inferred/")
    print("Done")

if __name__ == '__main__':
    tf.app.run()
