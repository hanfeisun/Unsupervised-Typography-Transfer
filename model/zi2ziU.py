# Unsupervised version of zi2zi

# Download the pretrained model from https://drive.google.com/file/d/0Bz6mX0EGe2ZuNEFSNWpTQkxPM2c/view
# then unzip it and put it under the project folder

import tensorflow as tf
import argparse

from zi2ziNet.unet import UNet

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', dest='Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', dest='fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=20,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=200,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', dest='flip_labels', type=int, default=None,
                    help='whether flip training data labels or not, in fine tuning')
parser.add_argument('--augment', dest='augment', action='store_true', default=False)
parser.add_argument('--init_step', dest='init_step', type=int, default=0)


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
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty,
                     )
        model.register_session(sess)
        model.build_model(is_training=True)
        tf.global_variables_initializer().run()
        inspect_graph()

        model.trainU(lr=args.lr, epoch=args.epoch,
                     schedule=args.schedule, fine_tune=None,
                     sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,
                     flip_labels=args.flip_labels,
                     freeze_encoder=args.freeze_encoder, augment=args.augment, model_dir=args.experiment_dir, init_step=args.init_step)


if __name__ == '__main__':
    tf.app.run()
