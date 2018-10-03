# -*- coding; utf-8 -*-

import tensorflow as tf
import numpy as np
import cvae
import os
import shutil
import time 
import tqdm 


class train_cvae_graph:
    
    def __init__(self, 
                dim_input=1440,
                dim_z=2, 
                dim_label=109, 
                n_epochs=100, 
                batch_size=256, 
                learn_rate=0.001, 
                n_hidden=[512, 256, 128],
                save_name=None,
                save_path=None,
                is_train=True,
                plot_prefix="./tensorboard/",
                model_prefix="./train_models/"): 

        tf.reset_default_graph()
        self.params = "Zdim_{0}_Nhid_{1}".format(dim_z, "-".join([str(i) for i in n_hidden]))
        
        if save_name:
            self.params += "_" + save_name

        self.plot_path = plot_prefix + save_path + "/train_{}".format(self.params)
        self.model_path = model_prefix + save_path + "/{}.ckpt".format(self.params)
        
        ## set cvae graph
        self.x_hat = tf.placeholder(tf.float32, shape=[None, dim_input], name='inputs')
        self.x = tf.placeholder(tf.float32, shape=[None, dim_input], name='targets')
        self.y = tf.placeholder(tf.float32, shape=[None, dim_label], name='input_labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train') if is_train else None

        self.x_, self.z, self.mu, self.sigma, self.loss, self.neg_marginal_likelihood, self.KL_divergence = cvae.cvae(self.x_hat, 
                                                                                         self.x, 
                                                                                         self.y, 
                                                                                         dim_input, 
                                                                                         dim_z, 
                                                                                         n_hidden, 
                                                                                         self.keep_prob,
                                                                                         self.phase_train)

        self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
        self.merged = tf.summary.merge_all()
        

    def train(self, X_train, X_labels, batch_size, epochs, keep_prob=1.):

        if os.path.exists(self.plot_path):
            print("plot data exists. Removed {}".format(self.plot_path))
            shutil.rmtree(self.plot_path)
        self.train_writer = tf.summary.FileWriter(self.plot_path)

        ## train graph
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={self.keep_prob:1.})

            for epoch in range(epochs):
                print("epoch:", epoch)
                st = time.time()
                
                rnd_ind = np.arange(len(X_train))
                np.random.shuffle(rnd_ind)
                rnd_X = X_train[rnd_ind]
                rnd_Y = X_labels[rnd_ind]
            
                for i in tqdm.tqdm(range(len(rnd_X)//batch_size)):
                    X = rnd_X[i*batch_size : (i+1)*batch_size]
                    Y = rnd_Y[i*batch_size : (i+1)*batch_size]

                    summary, _, total_loss, loss_likelihood, loss_divergence = sess.run(
                        (self.merged, self.train_op, self.loss, self.neg_marginal_likelihood, self.KL_divergence),
                        feed_dict={self.x_hat: X, self.x: X, self.y: Y, self.keep_prob: keep_prob, self.phase_train: True})
                    self.train_writer.add_summary(summary, epoch)
                
                end = time.time() 
                print("loss %03.2f LL %03.2f KLD %03.2f, time %03.2f" % (total_loss, -loss_likelihood, loss_divergence, end-st))

            self.train_writer.close()
            saver = tf.train.Saver()
            saver.save(sess, self.model_path)


    def evaluation(self, X_test, X_test_labels):
    
        klds = []
        config = tf.ConfigProto(device_count={"GPU":0})
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            for i,j in tqdm.tqdm(zip(X_test, X_test_labels), total=len(X_test)):
                klds.append(sess.run([self.KL_divergence],
                                     feed_dict={self.x_hat: i.reshape(1, -1),
                                                self.y: j.reshape(1, -1),
                                                self.keep_prob : 1}))

        return np.array(klds)


    def latent_vars(self, X_test, X_test_labels):
        zs = []
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            for i,j in tqdm.tqdm(zip(X_test, X_test_labels), total=len(X_test)):
                zs.append(sess.run(self.z,
                                    feed_dict={self.x_hat: i.reshape(1, -1),
                                                self.y: j.reshape(1, -1),
                                                self.keep_prob: 1.}))
            return np.array(zs)

    def reconst_error(self, X_test, X_test_labels):
    
        mlhs = []
        config = tf.ConfigProto(device_count={"GPU":0})
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            for i,j in tqdm.tqdm(zip(X_test, X_test_labels), total=len(X_test)):
                mlhs.append(sess.run([self.neg_marginal_likelihood],
                                     feed_dict={self.x: i.reshape(1, -1),
                                                self.x_hat: i.reshape(1, -1),
                                                self.y: j.reshape(1, -1),
                                                self.keep_prob : 1}))

        return np.array(mlhs)

    def reconst(self, X_test, X_test_labels):
    
        reconsts = []
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            for i,j in tqdm.tqdm(zip(X_test, X_test_labels), total=len(X_test)):
                reconsts.append(sess.run([self.x_],
                                     feed_dict={self.x_hat: i.reshape(1, -1),
                                                self.y: j.reshape(1, -1),
                                                self.keep_prob : 1}))

        return np.array(reconsts)
