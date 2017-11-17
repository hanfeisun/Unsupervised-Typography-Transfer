import tensorflow as tf
import argparse

from model.zi2ziNet.unet import UNet



def inspect_graph():
    from pprint import pprint
    var_list = tf.global_variables()
    collection_list = tf.get_default_graph().get_all_collection_keys()
    pprint(var_list)
    pprint(tf.get_default_graph().get_tensor_by_name("no_target_A_and_B_images:0"))


from subprocess import call


def main():
    with tf.Session() as sess:
        model = UNet()
        model.register_session(sess)
        model.build_model()
        inspect_graph()

    print("Done")


if __name__ == '__main__':
    main()
