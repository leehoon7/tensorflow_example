import gym
import tensorflow as tf
import numpy as np
from policy_network import Policy
import matplotlib.pyplot as plt
import time


def prepro(I, render = False):

    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """

    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # down-sample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1

    if render:
        return I

    return I.astype(np.float).ravel()


def check_img_diff():
    env = gym.make("Pong-v0")
    obs = env.reset()

    for i in range(50):
        env.render()
        obs, rew, done, info = env.step(5)
        time.sleep(0.1)

    plt.imshow(obs)
    plt.show()
    plt.close('all')

    plt.imshow(prepro(obs, True))
    plt.show()
    plt.close('all')


def main():
    env = gym.make("Pong-v0")
    obs = env.reset()
    obs = prepro(obs)
    obs = np.reshape(obs, [1, 6400])

    X = tf.placeholder(tf.float32, [None, 6400])
    X_img = tf.reshape(X, [-1, 80, 80, 1])
    # 32 filters : 3 * 3 * 1
    W1 = tf.Variable(tf.random_normal([8, 8, 1, 4], stddev=0.1))
    b1 = tf.Variable(tf.zeros([4]))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1 + b1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    W2 = tf.Variable(tf.random_normal([4, 4, 4, 8], stddev=0.1))
    b2 = tf.Variable(tf.zeros([8]))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2 + b2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W3 = tf.Variable(tf.random_normal([2, 2, 8, 16], stddev=0.1))
    b3 = tf.Variable(tf.zeros([16]))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3 + b3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.reshape(L3, [-1, 5 * 5 * 16])

    W4 = tf.Variable(tf.random_normal([5 * 5 * 16, 20], stddev=0.1))
    b4 = tf.Variable(tf.zeros([20]))
    L4 = tf.matmul(L3, W4)
    L4 = tf.nn.relu(L4 + b4)

    #W5 = tf.Variable(tf.)

    loss = L4[0][0] - 1
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        l1, l2, l3, l4, b, c = sess.run([L1, L2, L3, L4, loss, optimizer], feed_dict={X: obs})

        print(l1.shape)
        print(l2.shape)
        print(l3.shape)
        print(l4.shape)


if __name__ == '__main__':
    #check_img_diff()
    main()