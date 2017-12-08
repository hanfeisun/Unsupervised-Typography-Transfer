import tensorflow as tf
from pdb import set_trace

G_PREFIX = "generator"
D_PREFIX = "discriminator"


def model_fn(features, labels, mode, params):
    def conv(feature, num_outputs, kernel_size, stride, scope_name, reuse=False,
             padding='same', BN=True):
        # construct a convolutional layer group
        with tf.variable_scope(scope_name, reuse=reuse):
            conv_x = tf.layers.conv2d(feature, num_outputs,
                kernel_size=kernel_size, strides=stride, padding=padding,
                name='conv')
            if BN:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    conv_x = tf.layers.batch_normalization(conv_x,
                        training=True, name='batchnorm')
                else:
                    conv_x = tf.layers.batch_normalization(conv_x,
                        name='batchnorm')
            conv_x = tf.nn.elu(conv_x, name='elu')
        return conv_x

    def deconv(feature, num_outputs, kernel_size, stride, scope_name,
               reuse=False, padding='same', BN=True):
        # construct a deconvolutional layer group
        with tf.variable_scope(scope_name, reuse=reuse):
            deconv_x = tf.layers.conv2d_transpose(feature, num_outputs,
                kernel_size=kernel_size, strides=stride, padding=padding,
                name='deconv')
            if BN:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deconv_x = tf.layers.batch_normalization(deconv_x,
                        training=True, name='batchnorm')
                else:
                    deconv_x = tf.layers.batch_normalization(deconv_x,
                        name='batchnorm')
        return deconv_x

    # data_format is (batch, height, width, channels)

    with tf.variable_scope(G_PREFIX):
        # architecture picture: https://www.dropbox.com/s/1xjzj7u1nf4x09k/IMG_0073.JPG?dl=0
        e1 = conv(features['source'], num_outputs=64, kernel_size=3, stride=1,
                  scope_name='e1')
        e2 = conv(e1, 64, 3, 2, 'e2')
        e3 = conv(e2, 128, 3, 1, 'e3')
        e4 = conv(e3, 128, 3, 2, 'e4')
        e5 = conv(e4, 256, 3, 1, 'e5')
        e6 = conv(e5, 256, 3, 2, 'e6')
        e7 = conv(e6, 512, 3, 1, 'e7')
        e8 = conv(e7, 512, 3, 2, 'e8')

        d1 = deconv(e8, 512, 3, 2, 'd1')
        d2 = conv(tf.concat([e7, d1], axis=3), 512, 3, 1, 'd2')
        d3 = conv(d2, 256, 3, 1, 'd3')
        d4 = deconv(d3, 256, 3, 2, 'd4')
        d5 = conv(tf.concat([e5, d4], axis=3), 256, 3, 1, 'd5')
        d6 = conv(d5, 128, 3, 1, 'd6')
        d7 = deconv(d6, 128, 3, 2, 'd7')
        d8 = conv(tf.concat([e3, d7], axis=3), 128, 3, 1, 'd8')
        d9 = conv(d8, 64, 3, 1, 'd9')
        d10 = deconv(d9, 64, 3, 2, 'd10', BN=False)
        d11 = conv(tf.concat([e1, d10], axis=3), 64, 3, 1, 'd11')
        d12 = conv(d11, 1, 3, 1, 'd12', BN=False)
        generated = tf.nn.sigmoid(d12, 'gen')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"s": features["source"], "t": features["target"], "g": generated})

    def discriminator(image, reuse=False):
        with tf.variable_scope(D_PREFIX, reuse=reuse):
            net = conv(image, 2, 3, 1, 'dis_conv1')
            net = conv(net, 4, 3, 2, 'dis_conv2')
            net = conv(net, 8, 3, 2, 'dis_conv3')
            net = conv(net, 16, 3, 2, 'dis_conv4')
            logits = tf.layers.dense(tf.contrib.layers.flatten(net), 1, name='logits')
            prob = tf.nn.sigmoid(logits, 'prob')
        return logits, prob

    real_target = features['target']
    fake_target = generated
    real_logits, real_prob = discriminator(real_target)
    fake_logits, fake_prob = discriminator(fake_target, reuse=True)

    # real samples should be classified as one
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                labels=tf.ones_like(real_logits)))
    # real samples should be classified as zero
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                labels=tf.zeros_like(fake_logits)))
    d_loss = d_loss_real + d_loss_fake

    d12_flatten = tf.reshape(d12, [-1, 64 * 64])
    target_flatten = tf.reshape(real_target, [-1, 64 * 64])

    # Use cross entropy as pixel loss.
    # L1 or L2 loss may also be considerable.
    pixel_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d12_flatten,
                                                labels=target_flatten))
    g_loss = pixel_loss - d_loss_fake

    ##### Construct e1 batch images
    def get_img(e):
        e1_img = get_row(e, 0)
        for grid_row in range(1, 8):
            row = get_row(e, grid_row)
            e1_img = tf.concat([e1_img, row], axis=0)
        e1_img = tf.reshape(e1_img, [512, 512, 1])
        return e1_img

    def get_row(e, grid_row):
        start = grid_row * 8
        row = e[start]
        for i in range(1, 8):
            row = tf.concat([row, e[start + i]], axis=1)
        return row

    e1_batch_imgs = []
    for i in range(8):
        e1_img = get_img(e1[i])
        e1_batch_imgs.append(e1_img)
    e1_batch_imgs = tf.convert_to_tensor(e1_batch_imgs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("d_fake_loss", d_loss_fake)
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("pixel_loss", pixel_loss)
        tf.summary.scalar("g_loss", g_loss)
        tf.summary.image("original", features['source'])
        tf.summary.image("target", features['target'])
        tf.summary.image("transfered", generated)
        tf.summary.image("conv1", e1_batch_imgs) ##### Added
    else:
        # EVAL
        tf.summary.scalar("cross_entropy_loss", pixel_loss)
        tf.summary.image("eval_original", features['source'], max_outputs=10)
        tf.summary.image("eval_target", features['target'], max_outputs=10)
        tf.summary.image("eval_transfered", generated, max_outputs=10)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = g_loss + d_loss
    else:
        # EVAL
        loss = pixel_loss

    lr = params['learning_rate']
    g_vars, d_vars = [], []
    for v in tf.trainable_variables():
        if v.name.startswith(G_PREFIX):
            g_vars.append(v)
        elif v.name.startswith(D_PREFIX):
            d_vars.append(v)
    g_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_vars,
                                                                   global_step=tf.train.get_global_step())
    d_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_vars,
                                                                   global_step=tf.train.get_global_step())

    train_logging_hook = tf.train.LoggingTensorHook(
        {"g_loss": g_loss, "d_loss": d_loss, "cross_entropy_loss": pixel_loss}, every_n_iter=100)
    train_summary_hook = tf.train.SummarySaverHook(
        save_steps=10,
        output_dir="./board",
        summary_op=tf.summary.merge_all())

    eval_logging_hook = tf.train.LoggingTensorHook({"cross_entropy_loss": loss}, every_n_iter=100)
    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=10,
        output_dir="./board_eval",
        summary_op=tf.summary.merge_all()
    )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=tf.group(g_train_op, d_train_op),
        eval_metric_ops=None,
        training_hooks=[train_logging_hook, train_summary_hook],
        evaluation_hooks=[eval_logging_hook, eval_summary_hook]
    )
