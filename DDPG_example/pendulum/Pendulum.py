import gym
import tensorflow as tf
import random
import numpy as np
from collections import deque
import time


def main():

    env = gym.make("Pendulum-v0")
    obs = env.reset()
    obs = obs.reshape(-1, 3)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    noise_epsilon = 1
    noise_epsilon_min = 0.01
    noise_epsilon_decay = 0.005

    replay_memory = deque(maxlen=5000)
    batch_size = 500
    gamma = 0.99
    episode = 500
    cur_episode = 1
    tau_ = 0.995

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001

    #print(state_dim, action_dim, action_bound)

    ''' make placeholders '''
    S = tf.placeholder(tf.float32, [None, state_dim])
    A = tf.placeholder(tf.float32, [None, action_dim])
    T = tf.placeholder(tf.float32, [None, 1])
    tau = tf.constant([0.995])


    ''' critic network '''
    C_W1 = tf.Variable(tf.random_normal([state_dim + action_dim, 64], stddev=0.1))
    C_b1 = tf.Variable(tf.zeros([64]))
    C_L1 = tf.nn.relu(tf.matmul(tf.concat([S, A], 1), C_W1) + C_b1)

    C_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    critic_output = tf.matmul(C_L1, C_W2)

    C_loss = tf.reduce_mean((T - critic_output) ** 2)
    C_optimizer = tf.train.AdamOptimizer(learning_rate=critic_learning_rate).minimize(C_loss)

    ''' critic target network '''
    CT_W1 = tf.Variable(tf.random_normal([state_dim + action_dim, 64], stddev=0.1))
    CT_b1 = tf.Variable(tf.zeros([64]))
    CT_L1 = tf.nn.relu(tf.matmul(tf.concat([S, A], 1), CT_W1) + CT_b1)

    CT_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    target_critic_output = tf.matmul(CT_L1, CT_W2)

    ''' actor network '''
    A_W1 = tf.Variable(tf.random_normal([state_dim, 64], stddev=0.1))
    A_b1 = tf.Variable(tf.zeros([64]))
    A_L1 = tf.nn.relu(tf.matmul(S, A_W1) + A_b1)

    A_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    actor_output = tf.nn.tanh(tf.matmul(A_L1, A_W2)) * action_bound

    A_loss1 = tf.nn.relu(tf.matmul(tf.concat([S, actor_output], 1), C_W1) + C_b1)
    A_loss = - tf.reduce_mean(tf.matmul(A_loss1, C_W2))
    A_optimizer = tf.train.AdamOptimizer(learning_rate=actor_learning_rate).minimize(A_loss, var_list=[A_W1, A_b1, A_W2])

    ''' actor target network '''
    AT_W1 = tf.Variable(tf.random_normal([state_dim, 64], stddev=0.1))
    AT_b1 = tf.Variable(tf.zeros([64]))
    AT_L1 = tf.nn.relu(tf.matmul(S, AT_W1) + AT_b1)

    AT_W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.1))
    target_actor_output = tf.nn.tanh(tf.matmul(AT_L1, AT_W2)) * action_bound

    ''' make a session'''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ''' update session '''
    up1 = AT_W1.assign(A_W1)
    up2 = AT_b1.assign(A_b1)
    up3 = AT_W2.assign(A_W2)
    up4 = CT_W1.assign(C_W1)
    up5 = CT_b1.assign(C_b1)
    up6 = CT_W2.assign(C_W2)

    sess.run([up1, up2, up3, up4, up5, up6])

    ''' tau update session '''
    t_up1 = AT_W1.assign(tau * AT_W1 + (1 - tau) * A_W1)
    t_up2 = AT_b1.assign(tau * AT_b1 + (1 - tau) * A_b1)
    t_up3 = AT_W2.assign(tau * AT_W2 + (1 - tau) * A_W2)
    t_up4 = CT_W1.assign(tau * CT_W1 + (1 - tau) * C_W1)
    t_up5 = CT_b1.assign(tau * CT_b1 + (1 - tau) * C_b1)
    t_up6 = CT_W2.assign(tau * CT_W2 + (1 - tau) * C_W2)

    ''' check actor & critic '''
    action = sess.run(actor_output, feed_dict={S: obs})
    critic_val = sess.run(critic_output, feed_dict={S: obs, A: action})
    target_action = sess.run(target_actor_output, feed_dict={S: obs})
    target_critic_val = sess.run(target_critic_output, feed_dict={S: obs, A: action})

    print(action, critic_val)
    print(target_action, target_critic_val)

    ''' episode data '''
    A_loss_epi = []
    C_loss_epi = []
    reward_epi = []
    current_time = time.time()

    for i in range(episode):

        done = False
        A_loss_list = []
        C_loss_list = []
        reward_list = []

        while not done:

            ''' action with gaussian noise for exploration '''
            ''' action should be in [-2, 2] --> clip action value '''
            action = sess.run(actor_output, feed_dict={S: obs})
            action = action[0] + noise_epsilon * np.random.normal(0, 0.5, 1)
            action = np.clip(action, -action_bound, action_bound)

            ''' save a state before do action '''
            bef_obs = obs

            ''' do action and get new state, reward, .. '''
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(-1, 3)

            ''' save transition to replay memory '''
            reward_list.append(reward)
            transition = [bef_obs[0], action, reward, obs[0], done]
            replay_memory.append(transition)

            ''' training process '''
            if len(replay_memory) >= batch_size:
                train_data = random.sample(replay_memory, batch_size)

                t_bef_state = np.stack([data[0] for data in train_data])
                t_action = np.array([data[1] for data in train_data])
                t_reward = np.array([data[2] for data in train_data])
                t_aft_state = np.stack([data[3] for data in train_data])
                t_terminal = np.array([data[4] for data in train_data])

                ''' compute target : 12 line in pseudocode '''
                action_next = sess.run(target_actor_output, feed_dict={S: t_aft_state})
                q_val = sess.run(target_critic_output, feed_dict={S: t_aft_state, A: action_next})
                q_val = q_val.squeeze()

                t_terminal = (t_terminal == False).astype(int)

                batch_target = t_reward + gamma * q_val * t_terminal
                batch_target = batch_target.reshape(-1, 1)

                ''' update Q-function : 13 line in pseudocode '''
                C_loss_val, _ = sess.run([C_loss, C_optimizer], feed_dict={S: t_bef_state, A: t_action, T: batch_target})
                C_loss_list.append(C_loss_val)

                ''' update policy : 14 line in pseudocode '''
                A_loss_val, _ = sess.run([A_loss, A_optimizer], feed_dict={S: t_bef_state})
                A_loss_list.append(A_loss_val)

                ''' update target networks with tau : 15 line in pseudocode '''
                sess.run([t_up1, t_up2, t_up3, t_up4, t_up5, t_up6])

            ''' render when episode over 1000 '''
            if cur_episode > 1000:
                env.render()

        ''' print episode information : loss, reward, ... etc. '''
        if len(replay_memory) > batch_size:

            C_loss_epi.append(sum(C_loss_list) / len(C_loss_list))
            A_loss_epi.append(sum(A_loss_list) / len(A_loss_list))
            reward_epi.append(sum(reward_list))

            print('***********************')
            print('episode : ', cur_episode)
            print('critic loss : ', C_loss_epi[-1])
            print('actor loss : ', A_loss_epi[-1])
            print('reward : ', reward_epi[-1])
            print('epsilon : ', noise_epsilon)
            print('time : ', time.time() - current_time)

        cur_episode += 1

        env.close()
        obs = env.reset()
        obs = obs.reshape(-1, 3)

        ''' decay epsilon for less exploration '''
        if noise_epsilon > noise_epsilon_min:
            noise_epsilon -= noise_epsilon_decay

    ''' final loss, reward data for plotting '''
    print(A_loss_epi)
    print(C_loss_epi)
    print(reward_epi)


if __name__ == '__main__':
    main()
