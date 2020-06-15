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
    print(env.action_space.n)

    policy = Policy()


if __name__ == '__main__':
    main()