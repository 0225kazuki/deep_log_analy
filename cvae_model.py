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


def cvae_encoder(x, y, n_hidden, n_output, keep_prob):
    with tf.variable_scope("cvae_encoder"):

        # inputs
        dim_y = int(y.get_shape()[1])
        input_tensor = tf.concat(axis=1, values=[x, y])
        
        # initializer
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        
        # params
        # 1st layer
        with tf.name_scope("w0"):
            w0 = tf.get_variable('w0', [input_tensor.get_shape()[1], n_hidden+dim_y], initializer=w_init)
            variable_summaries(w0)
        with tf.name_scope("b0"):
            b0 = tf.get_variable('b0', [n_hidden+dim_y], initializer=b_init)
            variable_summaries(b0)
        h0 = tf.matmul(input_tensor, w0) + b0
        with tf.name_scope("activated"):
            h0 = tf.nn.elu(h0)
            variable_summaries(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd layer
        with tf.name_scope("w1"):
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            variable_summaries(w1)
        with tf.name_scope("b1"):
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            variable_summaries(b1)
        h1 = tf.matmul(h0, w1) + b1
        with tf.name_scope("activated"):
            h1 = tf.nn.tanh(h1)
            variable_summaries(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        with tf.name_scope('wo'):
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=b_init)
            variable_summaries(wo)
        with tf.name_scope('bo'):
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            variable_summaries(b0)

        gaussian_params = tf.matmul(h1, wo) + bo

        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

        tf.summary.histogram('gaussian_mean', mean)
        tf.summary.histogram('gaussian_dev', stddev)

    return mean, stddev


def cvae_decoder(z, y, n_hidden, n_output, keep_prob, reuse=False):
    with tf.variable_scope('cvae_decoder', reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        
        # input
        input_tensor = tf.concat(axis=1, values=[z, y])

        # 1st layer
        with tf.name_scope("w0"):
            w0 = tf.get_variable('w0', [input_tensor.get_shape()[1], n_hidden], initializer=w_init)
            variable_summaries(w0)
        with tf.name_scope("b0"):
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            variable_summaries(b0)
        h0 = tf.matmul(input_tensor, w0) + b0
        with tf.name_scope('activated'):
            h0 = tf.nn.tanh(h0)
            variable_summaries(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd layer
        with tf.name_scope("w1"):
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            variable_summaries(w1)
        with tf.name_scope("b1"):
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            variable_summaries(b1)
        h1 = tf.matmul(h0, w1) + b1
        with tf.name_scope('activated'):
            h1 = tf.nn.elu(h1)
            variable_summaries(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output
        with tf.name_scope("wo"):
            wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=b_init)
            variable_summaries(wo)
        with tf.name_scope("bo"):
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            variable_summaries(bo)
        x_ = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return x_


def cvae(x_hat, x, y, dim_input, dim_z, n_hidden, keep_prob, mlh_rate=1.0, kld_rate=1.0):

    mu, sigma = cvae_encoder(x_hat, y, n_hidden, dim_z, keep_prob)

    # reperameterize
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    x_ = cvae_decoder(z, y, n_hidden, dim_input, keep_prob)
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

    # return x_, z, loss, -marginal_likelihood, KL_divergence
    return y, z, mu, sigma, loss, -marginal_likelihood, KL_divergence

def decoder(z, dim_input, n_hidden):
    x_ = decoder(z, n_hidden, dim_input, 1.0, reuse=True)

    return x_
