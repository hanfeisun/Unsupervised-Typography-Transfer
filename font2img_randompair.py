# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"
sample = False

def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


import logging


def _get_gb2312_characters():
    # equivalent to level 2 of GBK
    higher_range = range(0xb0, 0xf7 + 1)
    lower_range = range(0xa1, 0xfe + 1)
    for higher in higher_range:
        for lower in lower_range:
            encoding = (higher << 8) | lower
            try:
                yield encoding.to_bytes(2, byteorder='big').decode(encoding='gb2312')
            except UnicodeDecodeError:
                hex_literal = '0x' + ''.join('%02x' % byte for byte in encoding.to_bytes(2, byteorder='big'))
                logging.warning('Unable to decode %s with GB2312' % hex_literal)
                pass


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img


def draw_example(src_ch, dst_ch, src_font, dst_font, canvas_size, x_offset, y_offset, dst_x_offset, dst_y_offset, filter_hashes,
                 mode="L"):
    global sample
    dst_img = draw_single_char(dst_ch, dst_font, canvas_size, dst_x_offset, dst_y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(src_ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new(mode, (canvas_size * 2, canvas_size), (255, 255, 255))


    if not sample:
        dst_img.save("dst.png")
        src_img.save("src.png")
        sample = True

    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True, mode="L",
             target_x_offset=None, target_y_offset=None, target_char_size=None, overlap=1.0):
    if target_x_offset is None:
        target_x_offset = x_offset

    if target_y_offset is None:
        target_y_offset = y_offset

    if target_char_size is None:
        target_char_size = char_size

    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=target_char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))


    charset_size = len(charset)

    set_A_size = sample_count
    set_B_size = int((1.0 - overlap) * sample_count)
    set_C_size = set_A_size - set_B_size

    if set_A_size + set_B_size > charset_size:
        raise ValueError("charset size not large enough")

    print("charset size is %s, sample size is %s, overlap ratio is %s" %(charset_size, sample_count, overlap))

    charset_src = charset[:set_A_size]
    charset_dst = charset[:set_C_size] + charset[-set_B_size:]



    count = 0
    for src_char, dst_char in zip(charset_src, charset_dst):
        if count == sample_count:
            break
        e = draw_example(src_char,dst_char, src_font, dst_font, canvas_size, x_offset, y_offset, target_x_offset, target_y_offset,
                         filter_hashes, mode)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.png" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='source font x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='source font y_offset')
parser.add_argument('--tgt_x_offset', dest='tgt_x_offset', type=int, default=None, help='target font x offset')
parser.add_argument('--tgt_y_offset', dest='tgt_y_offset', type=int, default=None, help='target font y_offset')
parser.add_argument('--tgt_char_size', dest='tgt_char_size', type=int, default=None, help='target character size')

parser.add_argument('--sample_count', dest='sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample_dir', help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')
parser.add_argument('--mode', dest='mode', choices=["L", "RGB"], default="L", help='mode for image, RGB or L')
parser.add_argument('--overlap', dest='overlap', type=float, default=1.0, help='overlap ratio')

args = parser.parse_args()

if __name__ == "__main__":
    if args.charset in ['GB2312']:
        charset = list(_get_gb2312_characters())
    elif args.charset in ['CN', 'JP', 'KR', 'CN_T']:
        charset = locals().get("%s_CHARSET" % args.charset)

    else:
        charset = [c for c in open(args.charset).readline()[:-1]]

    if args.shuffle:
        np.random.shuffle(charset)
    font2img(args.src_font, args.dst_font, charset, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             args.sample_count, args.sample_dir, args.label, args.filter, args.mode, args.tgt_x_offset,
             args.tgt_y_offset, args.tgt_char_size, args.overlap)
