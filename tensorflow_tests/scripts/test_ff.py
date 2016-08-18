import tensorflow as tf
import sys
import numpy as np
import logging
logging.basicConfig(stream=sys.stderr)
logging.getLogger().setLevel(logging.DEBUG)

class Test:
    def test1(self):
        pred = [
            [2.011655, 1.782139],
            [2.033969, 2.768493],
            [1.991037, 2.785066],
            [2.009654, 1.803043],
            [2.014947, 3.776756]]
        labels = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.1],
            [0.0, 0.1],
            [1.0, 0.0]]

        pred = tf.Variable(pred)
        labels = tf.Variable(labels)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            sm = tf.nn.softmax(pred)
            logging.info(sess.run(sm))
            # f = tf.argmax(pred, 1)
            # s = tf.argmax(labels, 1)
            # r = tf.equal(f, s)
            # d = tf.cast(r, tf.float32)
            # a = tf.reduce_mean(d)
            # logging.info(sess.run(f))
            # logging.info(sess.run(s))
            # logging.info(sess.run(r))
            # logging.info(sess.run(d))
            # logging.info(sess.run(a))

Test().test1()
