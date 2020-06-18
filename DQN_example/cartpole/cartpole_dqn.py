import gym
import tensorflow as tf
import numpy as np
import time
import random
from collections import deque


def main():
    env = gym.make("CartPole-v0")

    obs = env.reset()
    obs = obs.reshape(-1, 4)

    epsilon = 1.0
    epsilon_min = 0.01
    decay_rate = 0.005
    replay_memory = deque(maxlen=500)
    batch_size = 5
    gamma = 0.99
    episode = 1

    X = tf.placeholder(tf.float32, [None, 4])
    target = tf.placeholder(tf.float32, [None, 1])
    act_index = tf.placeholder(tf.int32, [None, 2])

    W1 = tf.Variable(tf.random_normal([4, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([64, 2], stddev=0.1))
    L2 = tf.matmul(L1, W2)

    q_value_current = tf.expand_dims(tf.gather_nd(L2, act_index), -1)

    loss = tf.reduce_mean((target - q_value_current) ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_epi = []
    reward_epi = []
    current_time = time.time()

    for i in range(1000):

        done = False
        loss_list = []
        reward_list = []

        while not done:
            # render game screen
            if episode >= 950:
                env.render()
            action = None

            # 'random action' or get 'q-value from network'
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                policy = sess.run([L2], feed_dict={X: obs})
                action = np.argmax(np.array(policy))

            # save a state before do action
            bef_obs = obs

            # do action and get new state, reward and done
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(-1, 4)

            reward_list.append(reward)

            transition = [bef_obs[0], action, reward, obs[0], done]
            replay_memory.append(transition)

            if len(replay_memory) >= 5:
                train_data = random.sample(replay_memory, batch_size)

                bef_state   = [data[0] for data in train_data]
                action      = np.array([data[1] for data in train_data])
                reward      = np.array([data[2] for data in train_data])
                aft_state   = [data[3] for data in train_data]
                terminal    = np.array([data[4] for data in train_data])

                aft_state = np.stack(aft_state)
                bef_state = np.stack(bef_state)

                # find aft_state's q-value
                q_val = sess.run([L2], feed_dict={X: aft_state})

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

        loss_epi.append(sum(loss_list) / len(loss_list))
        reward_epi.append(sum(reward_list))

        print('***********************')
        print('episode : ', episode)
        print('loss : ', loss_epi[-1])
        print('reward : ', reward_epi[-1])
        print('epsilon : ', epsilon)
        print('time : ', time.time() - current_time)

        episode += 1

        env.close()
        obs = env.reset()
        obs = obs.reshape(-1, 4)

        if epsilon > epsilon_min:
            epsilon -= decay_rate

    print(loss_epi)
    print(reward_epi)


if __name__ == '__main__':
    main()