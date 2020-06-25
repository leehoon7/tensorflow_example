import gym
import tensorflow as tf
import numpy as np
import time
import random
from collections import deque
import matplotlib.pyplot as plt


def main():
    #env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v0")

    state_dim = 4
    action_dim = 2

    obs = env.reset()
    obs = obs.reshape(-1, state_dim)

    epsilon = 1.0
    epsilon_min = 0.05
    decay_rate = 0.01
    replay_memory = deque(maxlen=1000)
    batch_size = 102
    gamma = 0.99
    episode = 1

    env.seed(0)
    target_replace_iter = 20
    tf.reset_default_graph()

    tf_state = tf.placeholder(tf.float32, [None, state_dim])
    tf_reward = tf.placeholder(tf.float32, [None, ])
    # target = tf.placeholder(tf.float32, [None, 1])
    tf_act_index = tf.placeholder(tf.int32, [None, 2])
    tf_state_n = tf.placeholder(tf.float32, [None, state_dim])
    tf_terminal = tf.placeholder(tf.float32, [None, ])

    with tf.variable_scope('q'):
        W1 = tf.Variable(tf.random_normal([state_dim, 512], stddev=0.1))
        b1 = tf.Variable(tf.zeros([512]))
        L1 = tf.nn.relu(tf.matmul(tf_state, W1) + b1)

        W2 = tf.Variable(tf.random_normal([512, 512], stddev=0.1))
        b2 = tf.Variable(tf.zeros([512]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

        W3 = tf.Variable(tf.random_normal([512, 512], stddev=0.1))
        b3 = tf.Variable(tf.zeros([512]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

        W4 = tf.Variable(tf.random_normal([512, action_dim], stddev=0.1))
        q_val = tf.matmul(L3, W4)

    q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')

    q_value_current = tf.expand_dims(tf.gather_nd(q_val, tf_act_index), -1)

    with tf.variable_scope('q_next'):
        W1_n = tf.Variable(tf.random_normal([state_dim, 512], stddev=0.1), trainable=False)
        b1_n = tf.Variable(tf.zeros([512]), trainable=False)
        L1_n = tf.nn.relu(tf.matmul(tf_state_n, W1_n) + b1_n)

        W2_n = tf.Variable(tf.random_normal([512, 512], stddev=0.1), trainable=False)
        b2_n = tf.Variable(tf.zeros([512]), trainable=False)
        L2_n = tf.nn.relu(tf.matmul(L1_n, W2_n) + b2_n)

        W3_n = tf.Variable(tf.random_normal([512, 512], stddev=0.1), trainable=False)
        b3_n = tf.Variable(tf.zeros([512]), trainable=False)
        L3_n = tf.nn.relu(tf.matmul(L2_n, W3_n) + b3_n)

        W4_n = tf.Variable(tf.random_normal([512, action_dim], stddev=0.1), trainable=False)
        q_val_next = tf.matmul(L3_n, W4_n)

        # q_value_next = tf.expand_dims(tf.gather_nd(L3, act_index), -1)

    target = tf_reward + gamma * tf.reduce_max(q_val_next, axis=1) * tf_terminal

    loss = tf.reduce_mean((target - q_value_current) ** 2)
    l2_reg = 1e-6
    for var in q_network_vars:
        if not 'bias' in var.name:
            loss += l2_reg * 0.5 * tf.nn.l2_loss(var)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_epi = []
    reward_epi = []
    current_time = time.time()

    for i in range(1000):

        done = False
        loss_list = []
        reward_list = []
        step = 0

        while not done:
            # render game screen
            #if episode % 100 == 1:
            #    env.render()
            action = None

            # 'random action' or get 'q-value from network'
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                policy = sess.run([q_val], feed_dict={tf_state: obs})
                action = np.argmax(np.array(policy))

            # save a state before do action
            bef_obs = obs

            #action = int(input('action : '))

            # do action and get new state, reward and done
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(-1, state_dim)

            reward_list.append(reward)

            transition = [bef_obs[0], action, reward, obs[0], done]
            replay_memory.append(transition)
            #print(action, reward, done)

            if len(replay_memory) >= batch_size:
                train_data = random.sample(replay_memory, batch_size)

                bef_state = [data[0] for data in train_data]
                action = np.array([data[1] for data in train_data])
                reward = np.array([data[2] for data in train_data])
                aft_state = [data[3] for data in train_data]
                terminal = np.array([data[4] for data in train_data])

                aft_state = np.stack(aft_state)
                bef_state = np.stack(bef_state)

                # # find aft_state's q-value
                # q_val = sess.run([L3], feed_dict={X: aft_state})
                #
                # # find maximum q-value
                # q_val = np.max(q_val, -1)[0]

                terminal = (terminal == False).astype(int)

                # set batch target value : r + gamma * max(q-value)
                # batch_target = reward + gamma * q_val * terminal
                # batch_target = batch_target.reshape(-1, 1)

                index = []
                for idx, action_idx in enumerate(action):
                    index.append([idx, action_idx])

                loss_val, _ = sess.run([loss, optimizer],
                                       feed_dict={tf_state: bef_state,
                                                  tf_reward: reward,
                                                  tf_act_index: index,
                                                  tf_state_n: aft_state,
                                                  tf_terminal: terminal})
                loss_list.append(loss_val)

            if step % target_replace_iter == 0:
                t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
                e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
                sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

            step += 1

        reward_epi.append(sum(reward_list))
        print('reward : ', reward_epi[-1])

        if len(replay_memory) >= batch_size:
            loss_epi.append(sum(loss_list) / len(loss_list))
            reward_epi.append(sum(reward_list))

            print('***********************')
            print('episode : ', episode)
            print('loss : ', loss_epi[-1])
            print('reward : ', reward_epi[-1])
            print('epsilon : ', epsilon)
            print('time : ', time.time() - current_time)


            data = reward_epi

            plt.plot(data)
            plt.title('reward')
            plt.savefig('reward1.png')
            plt.close('all')

            data = loss_epi

            plt.plot(data)
            plt.title('loss')
            plt.savefig('loss1.png')
            plt.close('all')

        episode += 1

        env.close()
        obs = env.reset()
        obs = obs.reshape(-1, state_dim)

        if epsilon > epsilon_min:
            epsilon -= decay_rate




if __name__ == '__main__':
    main()
