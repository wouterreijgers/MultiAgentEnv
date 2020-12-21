import math
import random
import torch

from ray.rllib import MultiAgentEnv

from env.hunter_env import HunterEnv
from env.multi_prey_env import MultiPreyEnv
from env.prey_env import PreyEnv
import numpy as np

from simulator.simulation import Simulation


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


class MultiPreyHunterEnv(MultiAgentEnv):
    def __init__(self, config):
        # Config the hunters
        num = config["num_hunters"]
        self.config = config
        self.agents = [HunterEnv(config) for _ in range(num)]
        self.index_map = {agent.type + "_" + str(i): i for i, agent in enumerate(self.agents)}
        self.dones = set()
        self.observation_space_hunter = self.agents[0].observation_space
        self.action_space_hunter = self.agents[0].action_space

        # Config the preys
        num = config["num_preys"]
        for i in range(num):
            self.agents.append(PreyEnv(config))
            self.index_map[self.agents[len(self.agents) - 1].type + "_" + str(i)] = len(self.agents) - 1
        self.observation_space_prey = self.agents[config["num_hunters"]].observation_space
        self.action_space_prey = self.agents[config["num_hunters"]].action_space
        self.reset_index_map = self.index_map.copy()

        self.hunter_wait = []
        self.prey_wait = []
        self.training = config['training']
        #print(self.training)
        # Build the visual simulation
        if not self.training:
            self.simulator = Simulation(config)
            self.time = 0
            self.episode_end = False

    def reset(self):
        #print("reset")
        self.dones = set()
        self.time = 0
        self.episode_end = False
        while len(self.agents) != len(self.reset_index_map):
            temp = self.agents.pop()
            if temp.type == "hunter":
                self.hunter_wait.append(temp)
            elif temp.type == "prey":
                self.prey_wait.append(temp)
        self.index_map = self.reset_index_map.copy()
        return {i: self.agents[a].reset() for i, a in self.reset_index_map.items()}

    def step(self, action_dict):
        #print("step")
        self.time += 1
        self.action_dict_copy = action_dict.copy()
        # print(action_dict)
        hunter_loc = None
        prey_loc = None
        amount_of_hunters_living = 0
        amount_of_preys_living = 0
        amount_of_hunters_total = 0
        amount_of_preys_total = 0
        """
        Find the positions of every living agent, these are needed in order to calculate the closest prey/hunter
        """
        #print(self.agents, self.index_map)
        for i, action in action_dict.items():
            if self.agents[self.index_map[i]].type == "hunter":
                amount_of_hunters_total += 1
                if not self.agents[self.index_map[i]].done:
                    amount_of_hunters_living += 1
                    if hunter_loc is None:
                        hunter_loc = self.agents[self.index_map[i]].get_position().unsqueeze(0)
                    else:
                        hunter_loc = torch.cat((hunter_loc, self.agents[self.index_map[i]].get_position().unsqueeze(0)),
                                               dim=0)
            elif self.agents[self.index_map[i]].type == "prey":
                amount_of_preys_total += 1
                if not self.agents[self.index_map[i]].done:
                    amount_of_preys_living += 1
                    if prey_loc is None:
                        prey_loc = self.agents[self.index_map[i]].get_position().unsqueeze(0)
                    else:
                        prey_loc = torch.cat((prey_loc, self.agents[self.index_map[i]].get_position().unsqueeze(0)),
                                             dim=0)

        """
        Find the closest prey/hunter and perform the 'step' function in the HunterEnv and PreyEnv
        """
        # print(action_dict)
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            dist = [self.config["sim"]["width"], self.config["sim"]["height"]]

            if "hunter" in i and amount_of_preys_living > 0:
                dist = find_closest(self.agents[self.index_map[i]].get_position(), prey_loc)

            if "prey" in i and amount_of_hunters_living > 0:
                dist = find_closest(self.agents[self.index_map[i]].get_position(), hunter_loc)
            obs[i], rew[i], done[i], info[i] = self.agents[self.index_map[i]].step(action, dist)
            if "prey" in i:
                rew[i] =0
            if amount_of_hunters_living == 0:
                done[i] = True
            if done[i]:
                self.dones.add(self.index_map[i])

            """
            Reproduce if the action allows it. First check if there are objects that can be reused, else make a 
            new one. after creation they also get added to all the lists needed.
            """
            if info[i]["reproduce"] and amount_of_hunters_living > 0 and not done[i]:
                if self.agents[self.index_map[i]].type == "hunter":
                    if len(self.hunter_wait) > 0:
                        self.agents.append(self.hunter_wait.pop())
                    else:
                        self.agents.append(HunterEnv(self.config))
                    id = self.agents[len(self.agents) - 1].type + "_" + str(amount_of_hunters_total)
                    n = 0
                    while id in self.index_map:
                        n += 1
                        id = self.agents[len(self.agents) - 1].type + "_" + str(amount_of_preys_total + n)
                    amount_of_hunters_total += 1
                elif self.agents[self.index_map[i]].type == "prey":
                    # print("new_prey", amount_of_preys_total)
                    if len(self.prey_wait) > 0:
                        self.agents.append(self.prey_wait.pop())
                    else:
                        self.agents.append(PreyEnv(self.config))
                    id = self.agents[len(self.agents) - 1].type + "_" + str(amount_of_preys_total)
                    n = 0
                    while id in self.index_map:
                        n += 1
                        id = self.agents[len(self.agents) - 1].type + "_" + str(amount_of_preys_total + n)
                    amount_of_preys_total += 1
                if id in self.index_map:
                    print("ERROR: ID ", id , " already in ", self.index_map)
                self.index_map[id] = len(self.agents) - 1
                obs[id] = self.agents[self.index_map[id]].reset()
                rew[id] = 1
                done[id] = False
                info[id] = {}
                # print(obs)
            if "hunter" in i:
                rew[i] = amount_of_hunters_living


        """
        Check if there are still some hunters, if not all the preys need to be killed otherwise it creates an error.
        """
        if amount_of_hunters_living == 0:
            for agent in self.agents:
                agent.done = True
        done["__all__"] = amount_of_hunters_living == 0
        self.episode_end = done["__all__"]
        # print(obs)
        return obs, rew, done, info

    def render(self):
        """
        Render the screen with PyGame
        TODO: For some reason 2 screens open...
        :return:
        """
        if not self.training:
            hunter_loc = None
            prey_loc = None
            amount_of_hunters_living = 0
            amount_of_preys_living = 0
            amount_of_hunters_total = 0
            amount_of_preys_total = 0
            for agent in self.agents:
                if agent.type == "hunter":
                    amount_of_hunters_total += 1
                    if not agent.done:
                        amount_of_hunters_living += 1
                        if hunter_loc is None:
                            hunter_loc = agent.get_position().unsqueeze(0)
                        else:
                            hunter_loc = torch.cat((hunter_loc, agent.get_position().unsqueeze(0)),
                                                   dim=0)
                elif agent.type == "prey":
                    amount_of_preys_total += 1
                    if not agent.done:
                        amount_of_preys_living += 1
                        if prey_loc is None:
                            prey_loc = agent.get_position().unsqueeze(0)
                        else:
                            prey_loc = torch.cat((prey_loc, agent.get_position().unsqueeze(0)),
                                                 dim=0)
            if hunter_loc is None:
                hunter_loc = []
            if prey_loc is None:
                prey_loc = []
            print(self.time)
            self.simulator.render(hunter_loc, prey_loc, self.time, amount_of_hunters_living, amount_of_hunters_total,
                                  amount_of_preys_living, amount_of_preys_total, self.episode_end)

    def close(self):
        print("render")
