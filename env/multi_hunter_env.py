import math
import random

import gym
from ray.rllib import MultiAgentEnv

from env.hunter_env import HunterEnv
from env.multi_prey_env import MultiPreyEnv


def find_closest(my_pos, target_pos):
    """
    Note, this function always returns the distance to the prey/hunter, never the direction -> meaning that the value
    of 'dist' will always be positive :param my_pos: :param target_pos: :return:
    """
    # print(my_pos, target_pos)
    dist_x = 400
    dist_y = 400
    dist_min = 400
    for x, y in target_pos:
        if abs(my_pos[0] - x) + abs(my_pos[1] - y) < dist_min:
            dist_x = my_pos[0] - x
            dist_y = my_pos[1] - y
            dist_min = abs(my_pos[0] - x) + abs(my_pos[1] - y)

    if dist_x > 200 or dist_y > 200:
        print(my_pos, target_pos)
    return [dist_x, dist_y]



class MultiHunterEnv(gym.Env):
    """
    By training this model you only train the hunters.
    The preys will behave randomly
    """
    def __init__(self, config):
        num = config.pop("num_hunters", 1)
        self.agents = [HunterEnv(config) for _ in range(num)]
        self.dones = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.multi_preys_env = MultiPreyEnv(config)
        self.preys = self.multi_preys_env.reset()

    def reset(self):
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        """
        Make the agents perform a step, notice that first we let the preys take steps
        :param action_dict:
        :return:
        """
        action_batch_t = {}
        for i in range(len(self.preys)):
            action_batch_t[i] = random.randint(0, 3)
        obs2, rew2, done2, info2 = self.multi_preys_env.step(action_batch_t)
        action = {}
        action[0] = action_dict

        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action.items():
            dist = [10,10]
            #if "hunter" in i: #and amount_of_preys_living > 0:
            dist = find_closest(self.agents[i].get_position(), self.multi_preys_env.get_pos())
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action, dist)
            # rew[i] = 1
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs[0][0], rew[0], done[0], info[0]


