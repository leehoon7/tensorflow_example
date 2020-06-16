import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque


# pre-process 210 x 160 x 3 frame into 6400 (80x80) 1D float vector
def prepro(I, render = False):

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
    epsilon = 1.0
    epsilon_min = 0.01
    decay_rate = 0.005
    replay_memory = deque(maxlen=50000)
    batch_size = 40
    gamma = 0.99
    episode = 1

    # pre-process observation
    obs = prepro(obs)
    obs = np.reshape(obs, [1, 6400])

    # set placeholder
    X = tf.placeholder(tf.float32, [None, 6400])
    X_img = tf.reshape(X, [-1, 80, 80, 1])
    target = tf.placeholder(tf.float32, [None, 1])
    act_index = tf.placeholder(tf.int32, [None, 2])

    # 1st layer : 4 filters, 8 * 8 kernel, 1 input channel
    W1 = tf.Variable(tf.random_normal([4, 4, 1, 4], stddev=0.1))
    b1 = tf.Variable(tf.zeros([4]))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1 + b1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W11 = tf.Variable(tf.random_normal([4, 4, 4, 8], stddev=0.1))
    b11 = tf.Variable(tf.zeros([8]))
    L11 = tf.nn.conv2d(L1, W11, strides=[1, 1, 1, 1], padding='SAME')
    L11 = tf.nn.relu(L11 + b11)
    L11 = tf.nn.max_pool(L11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2nd layer : 8 filters, 4 * 4 kernel, 4 input channels
    W2 = tf.Variable(tf.random_normal([4, 4, 8, 16], stddev=0.1))
    b2 = tf.Variable(tf.zeros([16]))
    L2 = tf.nn.conv2d(L11, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2 + b2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3rd layer : 16 filters, 2 * 2 kernel, 8 input channels
    W3 = tf.Variable(tf.random_normal([2, 2, 16, 16], stddev=0.1))
    b3 = tf.Variable(tf.zeros([16]))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3 + b3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # reshape : 2D -> 1D
    L3 = tf.reshape(L3, [-1, 5 * 5 * 16])

    # 4th layer : neural network, 400 -> 20
    W4 = tf.Variable(tf.random_normal([5 * 5 * 16, 20], stddev=0.1))
    b4 = tf.Variable(tf.zeros([20]))
    L4 = tf.matmul(L3, W4)
    L4 = tf.nn.relu(L4 + b4)

    # 5th layer : linear mapping, 20 -> 6
    W5 = tf.Variable(tf.random_normal([20, 6], stddev=0.1))
    b5 = tf.Variable(tf.zeros([6]))
    L5 = tf.matmul(L4, W5)
    L5 = L5 + b5

    # policy to use
    q_value = L5[0]
    q_value_batch = L5

    q_value_current = tf.expand_dims(tf.gather_nd(L5, act_index), -1)

    # loss to minimize
    loss = tf.reduce_mean((target - q_value_current) ** 2)
    #loss = L5[0][0] - 1
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(loss)

    # make a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_epi = []
    reward_epi = []
    for i in range(2000):

        done = False
        loss_list = []
        reward_list = []
        while not done:
            # render game screen
            if episode >= 180:
                env.render()
            action = None

            # 'random action' or get 'q-value from network'
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                policy = sess.run([q_value], feed_dict={X: obs})
                action = np.argmax(np.array(policy))

            # save a state before do action
            bef_obs = obs

            # do action and get new state, reward and done
            obs, reward, done, _ = env.step(action)

            real_done = done
            if reward == -1.0:
                real_done = True

            reward_list.append(reward)
            obs = prepro(obs)
            obs = np.reshape(obs, [1, 6400])

            transition = [bef_obs[0], action, reward, obs[0], real_done]
            replay_memory.append(transition)

            if len(replay_memory) >= 100:
                train_data = random.sample(replay_memory, batch_size)

                bef_state   = [data[0] for data in train_data]
                action      = np.array([data[1] for data in train_data])
                reward      = np.array([data[2] for data in train_data])
                aft_state   = [data[3] for data in train_data]
                terminal    = np.array([data[4] for data in train_data])

                aft_state = np.stack(aft_state)
                bef_state = np.stack(bef_state)

                # find aft_state's q-value
                q_val = sess.run([q_value_batch], feed_dict={X: aft_state})

                # find maximum q-value
                q_val = np.max(q_val, -1)[0]

                terminal = (terminal == False).astype(int)

                # set batch target value : r + gamma * max(q-value)
                batch_target = reward + gamma * q_val * terminal
                batch_target = batch_target.reshape(-1, 1)


                index = []
                for idx, action_idx in enumerate(action):
                    index.append([idx, action_idx])

                loss_val, _ = sess.run([loss, optimizer], feed_dict={X: bef_state, target: batch_target, act_index: index})
                loss_list.append(loss_val)


        print('***********************')
        print('episode : ', episode)
        print('loss : ', sum(loss_list)/len(loss_list))
        loss_epi.append(sum(loss_list)/len(loss_list))
        reward_epi.append(sum(reward_list))
        print('reward : ', reward_epi[-1])
        print('epsilon : ', epsilon)
        episode += 1

        env.close()
        obs = env.reset()
        obs = prepro(obs)
        obs = np.reshape(obs, [1, 6400])

        if epsilon > epsilon_min:
            epsilon -= decay_rate

    print(loss_epi)
    print(reward_epi)



if __name__ == '__main__':
    main()