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

    ''' make placeholders '''
    S = tf.placeholder(tf.float32, [None, state_dim])
    A = tf.placeholder(tf.float32, [None, action_dim])

    ''' actor network '''
    A_W1 = tf.Variable(tf.random_normal([state_dim, 64], stddev=0.1))
    A_b1 = tf.Variable(tf.zeros([64]))
    A_L1 = tf.nn.relu(tf.matmul(S, A_W1) + A_b1)

    A_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    actor_output = tf.nn.tanh(tf.matmul(A_L1, A_W2)) * action_bound

    ''' critic network '''
    C_W1 = tf.Variable(tf.random_normal([state_dim + action_dim, 64], stddev=0.1))
    C_b1 = tf.Variable(tf.zeros([64]))
    C_L1 = tf.nn.relu(tf.matmul(tf.concat([S, A], 1), C_W1) + C_b1)

    C_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    critic_output = tf.matmul(C_L1, C_W2)

    ''' make a session'''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ''' check actor & critic'''
    action = sess.run(actor_output, feed_dict={S: obs})
    critic_val = sess.run(critic_output, feed_dict={S: obs, A: action})
    print(action, critic_val)


if __name__ == '__main__':
    main()