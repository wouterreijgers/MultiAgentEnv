import time

import ray
import os
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dqn.dqn import DQNTrainer
from dqn.dqn_model import DQNModel
from dqn.hunter_model import DQNHunterModel
from dqn.hunter_policy import DQNHunterPolicy
from dqn.prey_model import DQNPreyModel
from dqn.prey_policy import DQNPreyPolicy
from env.hunter_env import HunterEnv
from env.multi_hunter_env import MultiHunterEnv
from env.multi_prey_hunter import MultiPreyHunterEnv


def env_creator(env_config):
    # return MultiHunterEnv(env_config)
    return MultiPreyHunterEnv(env_config)


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNPreyModel", DQNPreyModel)
    ModelCatalog.register_custom_model("DQNHunterModel", DQNHunterModel)

    env_config = {
        'num_hunters': 5,
        'num_preys': 10,
        'training': True,
        'hunters': {
            'start_amount': 5,
            'energy_to_reproduce': 30,
            'energy_per_prey_eaten': 10,
            'max_age': 20, },
        'preys': {
            'start_amount': 10,
            'birth_rate': 7,
            'max_age': 20},
        'sim': {
            'width': 10,
            'height': 10}
    }
    policy_config = {
        "hunter_policy_config": {
            "training": True,
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # "sample_batch_size": 50,
            "env": "MultiHunterEnv-v0",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-4,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            #"gamma": tune.grid_search([0.9983, 0.9985, 0.9986, 0.9987, 0.988, 0.989]),
            "epsilon": 1,
            "epsilon_decay": 0.9998,
            "epsilon_min": 0.1,
            "buffer_size": 20000,
            "batch_size": 2000,
            "env_config": env_config,
            "dqn_model": {
                "custom_model": "DQNHunterModel",
                "custom_model_config": {
                    "network_size": [32, 64, 128, 64, 32],
                },  # extra options to pass to your model
            },
        },
        "prey_policy_config": {
            "training": False,
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # "sample_batch_size": 50,
            "env": "MultiHunterEnv-v0",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-3,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            #"gamma": tune.grid_search([0.9983, 0.9985, 0.9986, 0.9987, 0.988, 0.989]),
            "epsilon": .9,
            "epsilon_decay": 0.99998,
            "epsilon_min": 0.01,
            "batch_size": 2000,
            "env_config": env_config,
            "dqn_model": {
                "custom_model": "DQNPreyModel",
                "custom_model_config": {

                    "network_size": [32, 64, 128, 64, 32],
                },  # extra options to pass to your model
            },
        }
    }



    env = register_env("MultiHunterEnv-v0", env_creator)
    test_env = MultiPreyHunterEnv(env_config)
    policies = {"hunter": (DQNHunterPolicy,
                           test_env.observation_space_hunter,
                           test_env.action_space_hunter,
                           policy_config["hunter_policy_config"]),
                "prey": (DQNPreyPolicy,
                         test_env.observation_space_prey,
                         test_env.action_space_prey,
                         policy_config["prey_policy_config"])}


    def policy_mapping_fn(agent_id):
        if "hunter" in agent_id:
            return "hunter"
        else:
            return "prey"

    config = {
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "torch",
        # "sample_batch_size": 50,
        "env": "MultiHunterEnv-v0",

        ########################################
        # Parameters Agent
        ########################################
        # "lr": 0.001,
        # #"lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
        # "gamma": 0.989,
        # "gamma": tune.grid_search([0.988, 0.989, 0.990, 0.992, 0.994, ]),
        "epsilon": 1,
        "epsilon_decay": 0.998,
        "epsilon_min": 0.01,
        "buffer_size": 80000,
        "batch_size": 1000,
        "env_config": env_config,
        #"model": "DQNHunterModel",
        "multiagent": {
            "policy_mapping_fn": policy_mapping_fn,
            "policies": policies,
            "policies_to_train": policies
        },
    }
    times = {}
    for i in range(20):
        timeBefore = time.time()
        tune.run(
            DQNTrainer,
            # checkpoint_freq=10,
            checkpoint_at_end=True,
            stop={"timesteps_total": i*10000},
            config=config,
        )
        timeAfter = time.time()
        times[i*10000] = timeAfter - timeBefore
    print("Total time taken: ", times)
