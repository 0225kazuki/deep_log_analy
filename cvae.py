# -*- coding; utf-8 -*-

import tensorflow as tf
import numpy as np


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)


def batch_norm_wrapper(inputs, phase_train=None, decay=0.99):
    epsilon = 1e-5
    out_dim = inputs.get_shape()[-1]
    scale = tf.Variable(tf.ones([out_dim]))
    beta = tf.Variable(tf.zeros([out_dim]))
    pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
    pop_var = tf.Variable(tf.ones([out_dim]), trainable=False)

    if phase_train == None:
       return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    rank = len(inputs.get_shape())
    # axes = range(rank - 1)  # nn:[0], conv:[0,1,2]
    axes = [0]

    batch_mean, batch_var = tf.nn.moments(inputs, axes)

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def update():  # Update ema.
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale, epsilon)
    def average():  # Use avarage of ema.
        train_mean = pop_mean.assign(ema.average(batch_mean))
        train_var = pop_var.assign(ema.average(batch_var))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon)

    return tf.cond(phase_train, update, average)


def cvae_encoder(x, y, n_hidden, n_output, keep_prob, phase_train=None):
    if type(n_hidden) != list:
        print("n_hidden should be list")
        return

    with tf.variable_scope("cvae_encoder"):

        # inputs
        dim_y = int(y.get_shape()[1])
        input_tensor = tf.concat(axis=1, values=[x, y])
        
        # initializer
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        
        # params
        # 1st layer
        for l in range(len(n_hidden)):
            with tf.name_scope("w%i" % (l)):
                w = tf.get_variable('w%i' % (l), [input_tensor.get_shape()[1], n_hidden[l]], initializer=w_init)
                variable_summaries(w)
            with tf.name_scope("b%i" % (l)):
                b = tf.get_variable('b%i' % (l), [n_hidden[l]], initializer=b_init)
                variable_summaries(b)
            h = tf.matmul(input_tensor, w) + b

            # bn = batch_norm_wrapper(h, phase_train)

            with tf.name_scope("activated"):
                h = tf.nn.elu(h)
                variable_summaries(h)
            h = tf.nn.dropout(h, keep_prob)

        # output layer
        with tf.name_scope('wo'):
            wo = tf.get_variable('wo', [h.get_shape()[1], n_output * 2], initializer=b_init)
            variable_summaries(wo)
        with tf.name_scope('bo'):
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            variable_summaries(bo)

        gaussian_params = tf.matmul(h, wo) + bo

        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

        tf.summary.histogram('gaussian_mean', mean)
        tf.summary.histogram('gaussian_dev', stddev)

    return mean, stddev


def cvae_decoder(z, y, n_hidden, n_output, keep_prob, reuse=False, phase_train=None):
    with tf.variable_scope('cvae_decoder', reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        
        # input
        input_tensor = tf.concat(axis=1, values=[z, y])

        for l in range(len(n_hidden)):
            # 1st layer
            with tf.name_scope("w%i" % l):
                w = tf.get_variable('w%i' % l, [input_tensor.get_shape()[1], n_hidden[-l]], initializer=w_init)
                variable_summaries(w)
            with tf.name_scope("b%i" % l):
                b = tf.get_variable('b%i' % l, [n_hidden[-l]], initializer=b_init)
                variable_summaries(b)
            h = tf.matmul(input_tensor, w) + b

            # h = batch_norm_wrapper(h, phase_train)

            with tf.name_scope('activated'):
                h = tf.nn.tanh(h)
                variable_summaries(h)
            h = tf.nn.dropout(h, keep_prob)

        # output
        with tf.name_scope("wo"):
            wo = tf.get_variable('wo', [h.get_shape()[1], n_output], initializer=b_init)
            variable_summaries(wo)
        with tf.name_scope("bo"):
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            variable_summaries(bo)
        x_ = tf.sigmoid(tf.matmul(h, wo) + bo)

    return x_


def cvae(x_hat, x, y, dim_input, dim_z, n_hidden, keep_prob, phase_train=None, mlh_rate=1.0, kld_rate=1.0):

    mu, sigma = cvae_encoder(x_hat, y, n_hidden, dim_z, keep_prob, phase_train)

    # reperameterize
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    x_ = cvae_decoder(z, y, n_hidden, dim_input, keep_prob, phase_train=phase_train)
    x_ = tf.clip_by_value(x_, 1e-8, 1 - 1e-8)

    # ELBO
    marginal_likelihood = tf.reduce_sum(x * tf.log(x_) + (1 - x) * tf.log(1 - x_), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = mlh_rate*marginal_likelihood - kld_rate*KL_divergence
    # ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('marginal_LH', marginal_likelihood)
    tf.summary.scalar('KLD', KL_divergence)

    return x_, z, mu, sigma, loss, -marginal_likelihood, KL_divergence


def decoder(z, dim_input, n_hidden):
    x_ = decoder(z, n_hidden, dim_input, 1.0, reuse=True)

    return x_
