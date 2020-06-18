import gym
import tensorflow as tf
import numpy as np
import time
import random
from collections import deque


def decode(i):

    out = [0] * 500
    out[i] = 1
    return np.array(out)


def main():
    env = gym.make("Taxi-v3")

    obs = env.reset()
    obs = decode(obs)

    epsilon = 1.0
    epsilon_min = 0.1
    decay_rate = 0.001
    replay_memory = deque(maxlen=500)
    batch_size = 50
    gamma = 0.999
    episode = 1

    S = tf.placeholder(tf.float32, [None, 500])
    A = tf.placeholder(tf.float32, [None, 6])

    target = tf.placeholder(tf.float32, [None, 1])

    # state layer
    W1 = tf.Variable(tf.random_normal([500, 30], stddev=0.1))
    b1 = tf.Variable(tf.zeros([30]))
    L1 = tf.nn.relu(tf.matmul(S, W1) + b1)

    # q-value layer
    SA = tf.concat([L1, A], 1)
    W3 = tf.Variable(tf.random_normal([30 + 6, 50], stddev=0.1))
    b3 = tf.Variable(tf.zeros([50]))
    L3 = tf.nn.relu(tf.matmul(SA, W3) + b3)

    W4 = tf.Variable(tf.random_normal([50, 30], stddev=0.1))
    b4 = tf.Variable(tf.zeros([30]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.Variable(tf.random_normal([30, 10], stddev=0.1))
    b5 = tf.Variable(tf.zeros([10]))
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

    W6 = tf.Variable(tf.random_normal([10, 1], stddev=0.1))
    q_value = tf.matmul(L5, W6)

    loss = tf.reduce_mean((target - q_value) ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_epi = []
    reward_epi = []
    current_time = time.time()

    action_set = []
    for i in range(6):
        temp = [0] * 6
        temp[i] = 1
        action_set.append(temp)

    action_set = np.array(action_set)

    for i in range(2000):

        done = False
        loss_list = []
        reward_list = []
        reward_to_print = None

        while not done:
            # render game screen
            if episode >= 980:
                env.render()
            action = None

            # 'random action' or get 'q-value from network'
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                feed_obs = np.broadcast_to(obs, (6, obs.shape[0]))
                policy = sess.run([q_value], feed_dict={S: feed_obs, A: action_set})
                action = np.argmax(np.array(policy))

            # save a state before do action
            bef_obs = obs

            # do action and get new state, reward and done
            obs, reward, done, _ = env.step(action)

            if done:
                reward_to_print = reward

            obs = decode(obs)
            reward_list.append(reward)

            transition = [bef_obs, action, reward, obs, done]
            replay_memory.append(transition)

            if len(replay_memory) >= batch_size:
                train_data = random.sample(replay_memory, batch_size)

                bef_state   = [data[0] for data in train_data]
                action      = np.array([data[1] for data in train_data])
                reward      = np.array([data[2] for data in train_data])
                aft_state   = [data[3] for data in train_data]
                terminal    = np.array([data[4] for data in train_data])

                #aft_state = np.stack(aft_state)
                bef_state = np.stack(bef_state)

                # find aft_state's q-value
                aft_state_feed = []
                for i in aft_state:
                    for _ in range(6):
                        aft_state_feed.append(i)

                aft_state_feed = np.stack(aft_state_feed)
                action_feed = np.array(action_set.tolist() * batch_size)

                q_val = sess.run(q_value, feed_dict={S: aft_state_feed, A: action_feed})
                q_val = q_val.reshape(-1, 6)

                #print(q_val)
                #print(np.max(q_val, -1))

                # find maximum q-value
                q_val = np.max(q_val, -1)

                terminal = (terminal == False).astype(int)

                # set batch target value : r + gamma * max(q-value)
                batch_target = reward + gamma * q_val * terminal
                batch_target = batch_target.reshape(-1, 1)
                #print(batch_target)
                #print(batch_target.shape)

                index = []
                for act in action:
                    temp = [0] * 6
                    temp[act] = 1
                    index.append(temp)
                #print(action)
                #print(index)
                #print(bef_state)

                loss_val, _ = sess.run([loss, optimizer], feed_dict={S: bef_state, target: batch_target, A: index})
                loss_list.append(loss_val)

                #print(loss_list)

        if len(replay_memory) >= batch_size:

            loss_epi.append(sum(loss_list) / len(loss_list))
            reward_epi.append(sum(reward_list))

            print('***********************')
            print('episode : ', episode)
            print('loss : ', loss_epi[-1])
            print('reward : ', reward_epi[-1])
            print('epsilon : ', epsilon)
            print('time : ', time.time() - current_time)
            print('final reward : ', reward_to_print)

        episode += 1

        env.close()
        obs = env.reset()
        obs = decode(obs)

        if epsilon > epsilon_min:
            epsilon -= decay_rate

    print(loss_epi)
    print(reward_epi)


if __name__ == '__main__':
    main()