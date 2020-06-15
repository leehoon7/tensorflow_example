import tensorflow as tf


class Policy():

    def __init__(self):
        self.lr = 0.001
        self.sess = tf.Session()
        self.obs = tf.placeholder(tf.float32, [None, 6400])