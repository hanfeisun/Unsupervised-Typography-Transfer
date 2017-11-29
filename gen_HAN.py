# -*- coding: utf-8 -*-


import tensorflow as tf
import argparse

from model.zi2ziNet.unet import UNet

parser = argparse.ArgumentParser(description='Train')
# parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
# parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
parser.add_argument('--text', dest='text', default=None)
# parser.add_argument('--checkpoint', dest='checkpoint', required=True)

args = parser.parse_args()


def inspect_graph():
    from pprint import pprint
    var_list = tf.global_variables()
    collection_list = tf.get_default_graph().get_all_collection_keys()
    pprint(var_list)
    pprint(tf.get_default_graph().get_tensor_by_name("no_target_A_and_B_images:0"))


from subprocess import call

X = None
def main(_):
    global X
    config = tf.ConfigProto()
    with open("./infer_charset", "w") as f:
        if args.text:
            f.write(args.text)
        f.write("南去經三國，東來過五湖。")
        f.write("︽永東國酬愛鬱靈鷹袋︾")
        f.write("あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモーリオ市、郊外のぎらぎらひかる草の波。")
        f.write("동해물과、백두산이、마르고닳도록、하느님이、보우하사。")
        f.write("ABCDEFGHIJKLM")
        f.write("\n")

    call(
        "rm -rf HAN_infer_sample && mkdir -p HAN_infer_sample",
        shell=True)
    call(
        "python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/XingKai.ttf --canvas_size 64  --sample_dir HAN_infer_sample \
	--char_size 48 --x_offset 0 --y_offset 0  --mode L  --charset GB2312 --tgt_x_offset 0 --tgt_y_offset 5 --tgt_char_size 60 --charset infer_charset",
        shell=True)

    call(
        "python3 img2pickle.py --dir HAN_infer_sample --save_dir HAN_infer_sample --split_ratio 1",
        shell=True
    )

    from model.HAN import model_fn
    from model.dataset import input_fn
    from model.io import dump_image

    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir="./model_dir/HAN", params={"learning_rate": 0.0002})
    transfers = nn.predict(input_fn=input_fn("./HAN_infer_sample/val.obj", shuffle=False, num_epochs=1))

    for i, t in enumerate(transfers):
        dump_image('./model_dir/%s.png' % i, t["g"])



if __name__ == '__main__':
    tf.app.run()
