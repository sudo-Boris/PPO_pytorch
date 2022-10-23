import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs) -> None:
        super(Agent, self).__init__()

        ### Parameters
        hidden_layer_size = 64

        ### Critic network
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    hidden_layer_size,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, 1), std=1.0),
        )

        ### Actor network
        #   init output layer (action) with similar weights
        #   so that actions have similar probabilities in the beginning
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    hidden_layer_size,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_layer_size, envs.single_action_space.n), std=0.01
            ),
        )

    def get_value(self, x):
        """Return the estimated value of the state

        Args:
            x (torch.Tensor): State tensor. Shape of observation space.

        Returns:
            Value: Estimated value of the state by the critic nextwork.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Return the action and value of the state

        Args:
            x (torch.Tensor): State tensor. Shape of observation space.
            action (torch.Tensor, optional): Action tensor. Shape of action space. Defaults to None.

        Returns:
            action (torch.Tensor): Sampled action for each environment.
            log_prob (torch.Tensor): Log probability of each action for each environment.
            entropy (torch.Tensor): Entropy of each action probability distribution for each environment.
            value (torch.Tensor): Value for each environment.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
