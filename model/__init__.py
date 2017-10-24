import tensorflow as tf


def model_fn(features, target, mode, params):
    def conv(feature, num_outputs, kernel_size, stride, padding):
        conv_x = tf.layers.conv2d(feature, num_outputs, kernel_size, stride, padding)
        conv_x = tf.layers.batch_normalization(conv_x)
        conv_x = tf.nn.elu(conv_x)
        return conv_x

    def deconv(feature, num_outputs, kernel_size, stride, padding):
        deconv_x = tf.layers.conv2d_transpose(features, num_outputs, kernel_size, stride, padding)
        return deconv_x

    # data_format is (batch, height, width, channels)

    e1 = conv(features['source'], 64, 3, 1, 'same')
    e2 = conv(e1, 64, 3, 2, 'valid')
    e3 = conv(e2, 128, 3, 1, 'same')
    e4 = conv(e3, 128, 3, 2, 'valid')
    e5 = conv(e4, 256, 3, 1, 'same')
    e6 = conv(e5, 256, 3, 2, 'valid')
    e7 = conv(e6, 512, 3, 1, 'same')
    e8 = conv(e7, 512, 3, 2, 'valid')

    d1 = deconv(e8, 512, 3, 2, 'valid')
    d2 = conv(tf.concat([e7, d1], axis=3), 512, 3, 1, 'same')
    d3 = conv(d2, 256, 3, 1, 'same')
    d4 = deconv(d3, 256, 3, 2, 'valid')
    d5 = conv(tf.concat([e5, d4], axis=3), 256, 3, 1, 'same')
    d6 = conv(d5, 128, 3, 1, 'same')
    d7 = deconv(d6, 128, 3, 2, 'valid')
    d8 = conv(tf.concat([e3, d7], axis=3), 128, 3, 1, 'same')
    d9 = conv(d8, 64, 3, 1, 'same')
    d10 = deconv(d9, 64, 3, 2, 'valid')
    d11 = conv(tf.concat([e1, d10], axis=3), 64, 3, 1, 'same')
    d12 = conv(d11, 1, 3, 1, 'same')

    dis1 = conv(d12, 2, 3, 1, 'same')
    dis2 = conv(dis1, 4, 3, 2, 'valid')
    dis3 = conv(dis2, 8, 3, 2, 'valid')
    dis4 = conv(dis3, 16, 3, 2, 'valid')
    flattened = tf.contrib.layers.flatten(dis4)


