import gym
import tensorflow as tf
import random
import numpy as np
from collections import deque


def main():

    env = gym.make("Pendulum-v0")
    obs = env.reset()
    obs = obs.reshape(-1, 3)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    noise_epsilon = 1
    replay_memory = deque(maxlen=500)

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
    #print(action, critic_val)

    for i in range(1000):

        done = False
        loss_list = []
        reward_list = []

        while not done:

            # action with gaussian noise for exploration
            action = sess.run(actor_output, feed_dict={S: obs})
            action = action[0] + noise_epsilon * np.random.normal(0, 0.5, 1)

            # save a state before do action
            bef_obs = obs

            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(-1, 3)

            reward_list.append(reward)
            transition = [bef_obs[0], action, reward, obs[0], done]



            env.render()

            #break

        env.close()
        obs = env.reset()
        obs = obs.reshape(-1, 3)
        break


if __name__ == '__main__':
    main()
