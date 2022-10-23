import torch


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, as_shape, device) -> None:
        ### Storage buffers
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + as_shape).to(device)
        self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
