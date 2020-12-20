# MultiAgentEnv
### Testing results
#### Config
During the training the config file is unchanged.
```markdown
env_config = {
        'num_hunters': 20,
        'num_preys': 100,
        'training': True,
        'hunters': {
            'start_amount': 20,
            'energy_to_reproduce': 30,
            'energy_per_prey_eaten': 10,
            'max_age': 20, },
        'preys': {
            'start_amount': 100,
            'birth_rate': 5,
            'max_age': 20},
        'sim': {
            'width': 200,
            'height': 200}
    }

stop={"timesteps_total": 2000},
```


#### Untrained model
This untrained model is almost the same as a full model, the difference is that the weights do not get changed. If chosen well it can result in some unexpected results.

| Trial name                                   | reward          |  episode_reward_max |   episode_reward_min |   episode_len_mean |
| -------------------------------------------- |:---------------:| :-----:| :-----:| :-----:|
| DQNAlgorithm_MultiHunterEnv-v0_5ddf4_00000   | 6907.03         |    10882 |                 4375 |            30.8594 |


#### Random model
The random model returns random values for both hunter and prey

| Trial name                                   | reward          |  episode_reward_max |   episode_reward_min |   episode_len_mean |
| -------------------------------------------- |:---------------:| :------------------:| :-------------------:| :-----------------:|
| DQNAlgorithm_MultiHunterEnv-v0_edc3c_00000   | 6322.61         |    10764            |                 4304 |            28.6522  |

### Features
#### Recycling object instances
The `env/multi_prey_hunter.py` file contains the MultiAgentEnv, in this multiagent env different `HunterEnv` and `PreyEnv` instances are build.
When running an episode the code will keep generating new Objects but as soon as we perform a `reset()` on the MultiAgentEnv all the new classes will be placed in a waiting queue. If a new hunter or prey is born we first try to recover that obj instance.