import tensorflow as tf


def model_fn(features, labels, mode, params):
    def conv(feature, num_outputs, kernel_size, stride, scope_name, reuse=False, padding='same'):
        with tf.variable_scope(scope_name, reuse=reuse):
            conv_x = tf.layers.conv2d(feature, num_outputs, kernel_size, stride, padding, name='conv')
            conv_x = tf.layers.batch_normalization(conv_x, name='batchnorm')
            conv_x = tf.nn.elu(conv_x, name='elu')
        return conv_x

    def deconv(feature, num_outputs, kernel_size, stride, scope_name, reuse=False, padding='same'):
        with tf.variable_scope(scope_name, reuse=reuse):
            deconv_x = tf.layers.conv2d_transpose(feature, num_outputs, kernel_size, stride, padding, name='deconv')
        return deconv_x

    # data_format is (batch, height, width, channels)

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
    d10 = deconv(d9, 64, 3, 2, 'd10')
    d11 = conv(tf.concat([e1, d10], axis=3), 64, 3, 1, 'd11')
    d12 = conv(d11, 1, 3, 1, 'd12')
    generated = tf.nn.sigmoid(d12, 'gen')

    def discriminator(image, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
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

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))

    d_loss = d_loss_real + d_loss_fake

    d12_flatten = tf.reshape(d12, [-1, 64 * 64])
    target_flatten = tf.reshape(real_target, [-1, 64 * 64])

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d12_flatten, labels=target_flatten))

    learning_rate = params['learning_rate']

    tf.summary.scalar("d_loss", d_loss)
    tf.summary.scalar("g_loss", g_loss)
    tf.summary.image("original", features['source'])
    tf.summary.image("target", features['target'])
    tf.summary.image("transfer", generated)
    loss = g_loss + d_loss * 0.05
    # loss = d_loss + g_loss
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    logging_hook = tf.train.LoggingTensorHook({"loss": loss, "g_loss": g_loss, "d_loss": d_loss}, every_n_iter=100)
    summary_hook = tf.train.SummarySaverHook(
        save_secs=5,
        output_dir="./board",
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook, summary_hook]
    )
