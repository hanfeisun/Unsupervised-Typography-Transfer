import tensorflow as tf

G_PREFIX = "generator"
D_PREFIX = "discriminator"


def model_fn(features, labels, mode, params):
    def conv(feature, num_outputs, kernel_size, stride, scope_name, reuse=False, padding='same', BN=True):
        # construct a convolutional layer group
        with tf.variable_scope(scope_name, reuse=reuse):
            conv_x = tf.layers.conv2d(feature, num_outputs, kernel_size, stride, padding, name='conv')
            if BN:
                conv_x = tf.layers.batch_normalization(conv_x, name='batchnorm',
                                                       training=(mode is tf.estimator.ModeKeys.TRAIN))
            conv_x = tf.nn.elu(conv_x, name='elu')
        return conv_x

    def deconv(feature, num_outputs, kernel_size, stride, scope_name, reuse=False, padding='same', BN=True):
        # construct a deconvolutional layer group
        with tf.variable_scope(scope_name, reuse=reuse):
            deconv_x = tf.layers.conv2d_transpose(feature, num_outputs, kernel_size, stride, padding, name='deconv')
            if BN:
                deconv_x = tf.layers.batch_normalization(deconv_x, name='batchnorm',
                                                         training=(mode is tf.estimator.ModeKeys.TRAIN)                                                        )
        return deconv_x

    # data_format is (batch, height, width, channels)

    with tf.variable_scope(G_PREFIX):
        # architecture picture: https://www.dropbox.com/s/1xjzj7u1nf4x09k/IMG_0073.JPG?dl=0
        e1 = conv(features['source'], 64, 3, 1, 'e1')
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
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    # real samples should be classified as zero
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    d12_flatten = tf.reshape(d12, [-1, 64 * 64])
    target_flatten = tf.reshape(real_target, [-1, 64 * 64])

    d_loss = d_loss_real + d_loss_fake

    # Use cross entropy as pixel loss.
    # L1 or L2 loss may also be considerable.
    pixel_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d12_flatten, labels=target_flatten))
    g_loss = pixel_loss - d_loss_fake

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("fake_loss", d_loss_fake)
        tf.summary.scalar("g_loss", g_loss)
        tf.summary.scalar("pixel_loss", pixel_loss)
        tf.summary.image("original", features['source'])
        tf.summary.image("target", features['target'])
        tf.summary.image("transfer_sigmoid", generated)

    else:
        # EVAL
        tf.summary.image("eval_original", features['source'], max_outputs=10)
        tf.summary.image("eval_target", features['target'], max_outputs=10)
        tf.summary.image("eval_transfer_sigmoid", generated, max_outputs=10)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = g_loss + d_loss
    else:
        # EVAL
        loss = pixel_loss

    theta_d = []
    theta_g = []
    for v in tf.trainable_variables():
        if v.name.startswith(D_PREFIX):
            theta_d.append(v)
        elif v.name.startswith(G_PREFIX):
            theta_g.append(v)

    learning_rate = params['learning_rate']
    d_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=theta_d,
                                                                           global_step=tf.train.get_global_step())
    g_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=theta_g,
                                                                           global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({"g_loss": g_loss, "d_loss": d_loss}, every_n_iter=100)
    summary_hook = tf.train.SummarySaverHook(
        save_steps=10,
        output_dir="./board",
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=tf.group(d_train, g_train),
        eval_metric_ops=None,
        training_hooks=[logging_hook, summary_hook]
    )
