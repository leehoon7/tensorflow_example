import tensorflow as tf
from collections import deque


class Policy():

    def __init__(self):
        self.lr = 0.001
        self.sess = tf.Session()
        self.obs = tf.placeholder(tf.float32, [None, 6400])
        self.memory = deque(maxlen=1000)

    def network(self):
        pass
        #self.model =