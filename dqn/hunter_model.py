from torch import nn, cat
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym.spaces import Discrete, Box


class DQNHunterModel(nn.Module, TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config
        self.name = name
        self.network_size = model_config["custom_model_config"]["network_size"]

        if isinstance(self.obs_space, Box):
            self.obs_shape = obs_space.shape[0]
        else:
            self.obs_shape = self.obs_space

        self.layers = nn.Sequential()
        last_size = self.obs_space.shape[0]
        i = 0
        for layer_size in self.network_size:
            self.layers.add_module("linear_{}".format(i), nn.Linear(last_size, layer_size))
            self.layers.add_module("relu_{}".format(i), nn.ReLU())
            last_size = layer_size
            i += 1
        self.layers.add_module("linear_{}".format(i), nn.Linear(last_size, num_outputs))
        
    @override(TorchModelV2)
    def forward(self, obs):
        return self.layers(obs)
