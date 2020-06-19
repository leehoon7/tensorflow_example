import gym
import tensorflow as tf
import random
import numpy as np


def main():

    env = gym.make("Pendulum-v0")
    obs = env.reset()
    obs = obs.reshape(-1, 3)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    act_learning_rate = 0.001
    print(state_dim, action_dim, action_bound)

    X = tf.placeholder(tf.float32, [None, state_dim])

    ''' actor network '''
    W1 = tf.Variable(tf.random_normal([state_dim, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    actor_output = tf.nn.tanh(tf.matmul(L1, W2)) * action_bound

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    action = sess.run(actor_output, feed_dict={X: obs})

    print(action)


if __name__ == '__main__':
    main()