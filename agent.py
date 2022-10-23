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
        self.hidden_layer_size = 64

        ### Critic network
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    self.hidden_layer_size,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_layer_size, self.hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_layer_size, 1), std=1.0),
        )

        ### Actor network
        #   init output layer (action) with similar weights
        #   so that actions have similar probabilities in the beginning
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    self.hidden_layer_size,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_layer_size, self.hidden_layer_size)),
            nn.Tanh(),
            layer_init(
                nn.Linear(self.hidden_layer_size, envs.single_action_space.n), std=0.01
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


class AtariAgent(nn.Module):
    def __init__(self, envs) -> None:
        super(AtariAgent, self).__init__()

        self.sharedNetwork_out = 512

        ### Detail 8: Shared feature extractor CNN
        #   The feature extractor is shared between the actor and critic networks.
        #   (4, 84, 84) -> (32, 20, 20) -> (64, 9, 9) -> (64, 7, 7)
        self.sharedNetwork = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, self.sharedNetwork_out)),
            nn.ReLU(),
        )

        self.actor = layer_init(
            nn.Linear(self.sharedNetwork_out, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(self.sharedNetwork_out, 1), std=1.0)

    def get_value(self, x):
        """Return the estimated value of the state

        Args:
            x (torch.Tensor): State tensor. Shape of observation space.

        Returns:
            Value: Estimated value of the state by the critic nextwork.
        """
        ### Detail 9: Scale input to [0, 1]
        #   Each pixel has a range of [0, 255].
        #   Scale the input to [0, 1] by deviding by 255.
        return self.critic(self.sharedNetwork(x / 255.0))

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
        ### Detail 9: Scale input to [0, 1]
        #   Each pixel has a range of [0, 255].
        #   Scale the input to [0, 1] by deviding by 255.
        hidden = self.sharedNetwork(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
