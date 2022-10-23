import random
import time
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils import parse_args, init_tb_writer
from env import make_envs
from agent import Agent
from rollout_buffer import RolloutBuffer


class PPO:
    def __init__(self, args):
        current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
        self.run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{current_time}"

        ### WandB init
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        #   Important: Init tensorboard writer AFTER wandb.init()
        self.writer = init_tb_writer(args, run_name=self.run_name)

        ### DO NOT MODIFY: Tracking setup
        self.global_step = 0
        self.start_time = time.time()
        self.num_updates = args.total_timesteps // args.batch_size
        self.num_steps = args.num_steps

        ### Learning setup
        #   Hyperparameters
        # --- Specific implementation details ---
        # Detail 1: vectorized environments
        # Detail 2: Different layer initializations (see agent.py)
        eps = 1e-5  # Detail 3: Change epsilon from pytorchs default 1e-8 to 1e-5
        self.anneal_lr = args.anneal_lr  # Detail 4: Learning rate annealing
        self.gae = args.gae  # Detail 5: Generalized Advantage Estimation
        self.gae_lambda = args.gae_lambda
        self.minibatch_size = (
            args.minibatch_size
        )  # Detail 6: sample one minibatch at a time
        self.norm_adv = args.norm_adv  # Detail 7: Advantage normalization
        self.clip_coef = (
            args.clip_coef
        )  # Detail 8: Clipping coefficient to clip policy loss
        self.clip_vloss = args.clip_vloss  # Detail 9: (Bool) Also clip value loss
        self.ent_coef = args.ent_coef  # Detail 10: Slow down entropy minimization. \
        # It helps exploration by limiting premature convergence to suboptimal policy.
        self.max_grad_norm = args.max_grad_norm  # Detail 11: Gradient clipping
        self.target_kl = args.target_kl  # Bonus detail: Early stopping
        # --- General hyperparameters ---
        self.num_envs = args.num_envs
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.update_epochs
        self.vf_coef = args.vf_coef
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        ### Create gym environments (vectorized) | Detail 1
        self.envs = make_envs(
            args.gym_id, args.seed, args.num_envs, args.capture_video, self.run_name
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        ### Get Agent, optimizer and rollout buffer
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=eps
        )
        self.rollout_buffer = RolloutBuffer(
            self.num_steps,
            self.num_envs,
            self.envs.single_observation_space.shape,
            self.envs.single_action_space.shape,
            device=self.device,
        )

    def learn(self):
        # num_updates also called episodes
        for update in range(1, self.num_updates + 1):
            if self.anneal_lr:  ### Detail 4: Learning rate annealing
                frac = 1.0 - (update - 1.0) / self.num_updates
                current_lr = self.learning_rate * frac
                self.optimizer.param_groups[0]["lr"] = current_lr

            self.policy_rollout()
            # print("Policy rollout done. Update")
            self.train()

    def train(self):

        ### Get batch to train on
        #   Also includes computation of advantages and returns
        (
            b_obs,
            b_actions,
            b_log_probs,
            b_advantages,
            b_returns,
            b_values,
        ) = self.get_flattened_batch()

        b_ins = np.arange(self.batch_size)

        ### Usefull stats for logging
        clipfracs = []  # How often was the policy clipped

        ### Update policy for K epochs
        for _ in range(self.num_epochs):
            np.random.shuffle(b_ins)
            ### Detail 6: sample one minibatch at a time
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_ins[start:end]

                ### predict values based on the minibatch
                _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )

                ### compute ratios
                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()

                ### Usefull stats for logging
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                ### Detail 7: Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std()
                        + 1e-8  # Adding a small value to avoid division by zero
                    )

                ### Detail 8: Clipping policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                # Paper takes the minimum of the two positive losses
                # Which is equivalent to taking the maximum of the two negative losses
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                ### Detail 9: Clipping value loss
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_values - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

                ### Overall loss
                #   Including detail 10: Slowing down entropy minimization
                #   Minimize policy loss and value loss, maximize entropy
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                ### Time for backpropagation
                self.optimizer.zero_grad()
                ### Detail 11: Gradient clipping
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                loss.backward()
                self.optimizer.step()

            ### Bonus Detail: Early stopping (on batch level)
            #   Also possible to implement on minibatch level
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        ### Usefull stats for logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        #   Is the value function a good indicator of the returns?
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ### DO NOT MODIFY: LOGGING
        self.writer.add_scalar(
            "charts/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.global_step,
        )
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), self.global_step
        )
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar(
            "losses/explained_variance", explained_var, self.global_step
        )
        print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar(
            "charts/SPS",
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step,
        )

    def policy_rollout(self):
        # self.rollout_buffer.reset()

        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)

        for step in range(self.num_steps):
            self.global_step += 1 * self.num_envs
            self.rollout_buffer.obs[step] = self.next_obs
            self.rollout_buffer.dones[step] = self.next_done

            with torch.no_grad():
                action, log_prob, _, value = self.agent.get_action_and_value(
                    self.next_obs
                )
                self.rollout_buffer.values[step] = value.flatten()

            self.rollout_buffer.actions[step] = action
            self.rollout_buffer.log_probs[step] = log_prob

            ### DO NOT MODIFY: Step the envs and log data
            #   (n_envs, obs_shape), (n_envs, ), (n_envs, ), (n_envs, Dict)
            self.next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            #   Sum up all rewards of the parallel environments
            self.rollout_buffer.rewards[step] = (
                torch.Tensor(reward).to(self.device).view(-1)
            )
            #   Transform to torch.Tensor
            self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(
                self.device
            ), torch.Tensor(done).to(self.device)

            ### Logging
            for item in info:
                if "episode" in item.keys():
                    self.writer.add_scalar(
                        "charts/episode_reward",
                        item["episode"]["r"],
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "charts/episode_length",
                        item["episode"]["l"],
                        self.global_step,
                    )
                    break

    def compute_gae(self, obs):
        """Compute Generalized Advantage Estimation (GAE)

        Args:
            obs (torch.Tensor): Next observations, that were not stored in the buffer

        Returns:
            returns (torch.Tensor): GAE computed returns
            advantages (torch.Tensor): GAE computed advantages
        """

        next_value = self.agent.get_value(obs).reshape(1, -1)
        advantages = torch.zeros_like(self.rollout_buffer.rewards).to(self.device)
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.rollout_buffer.dones[step]
                nextvalues = next_value
            else:
                next_non_terminal = 1.0 - self.rollout_buffer.dones[step + 1]
                nextvalues = self.rollout_buffer.values[step + 1]
            delta = (
                self.rollout_buffer.rewards[step]
                + self.gamma * nextvalues * next_non_terminal
                - self.rollout_buffer.values[step]
            )
            advantages[step] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        returns = advantages + self.rollout_buffer.values

        return returns, advantages

    def compute_returns(self, obs):
        """Compute standard returns and advantages.

        Args:
            obs (torch.Tensor): Next observations, that were not stored in the buffer.

        Returns:
            returns (torch.Tensor): Standard discounted sum of returns.
            advantages (torch.Tensor): Advantages, that are computed as returns - values. \\
                    How similar are the estimated returns (values) to the actual returns.
        """
        next_value = self.agent.get_value(obs).reshape(1, -1)
        returns = torch.zeros_like(self.rollout_buffer.rewards).to(self.device)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.rollout_buffer.dones[step]
                next_return = next_value
            else:
                next_non_terminal = 1.0 - self.rollout_buffer.dones[step + 1]
                next_return = returns[step + 1]
            returns[step] = (
                self.rollout_buffer.rewards[step]
                + self.gamma * next_non_terminal * next_return
            )
        advantages = returns - self.rollout_buffer.values

        return returns, advantages

    def get_flattened_batch(self):
        """Return flattened batch of observations, actions, log_probs, advantages, returns, and values.

        Return:
            obs (torch.Tensor): Flattened observations. Shape: (batch_size, obs_shape)
            actions (torch.Tensor): Flattened actions. Shape: (batch_size,)
            log_probs (torch.Tensor): Flattened log_probs. Shape: (batch_size,)
            advantages (torch.Tensor): Flattened advantages. Shape: (batch_size,)
            returns (torch.Tensor): Flattened returns. Shape: (batch_size,)
            values (torch.Tensor): Flattened values. Shape: (batch_size,)
        """

        # bootstrap values if not done
        with torch.no_grad():
            if self.gae:  ### Detail 5: GAE
                returns, advantages = self.compute_gae(self.next_obs)
            else:
                returns, advantages = self.compute_returns(self.next_obs)

        return (
            self.rollout_buffer.obs.reshape(
                (-1,) + self.envs.single_observation_space.shape
            ),
            self.rollout_buffer.actions.reshape(
                (-1,) + self.envs.single_action_space.shape
            ),
            self.rollout_buffer.log_probs.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            self.rollout_buffer.values.reshape(-1),
        )


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # DO NOT MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ### PPO with agent
    ppo = PPO(args)

    ### Start interaction with environment
    try:
        ppo.learn()
        ppo.envs.close()
        ppo.writer.close()
    except Exception or KeyboardInterrupt:
        print("Failed to run")
        ppo.envs.close()
        ppo.writer.close()

    # Stepping loop
    # try:
    #     obs = envs.reset()
    #     for _ in range(200):
    #         action = envs.action_space.sample()
    #         obs, reward, done, info = envs.step(action)
    #         # if done:
    #         for item in info:
    #             if "episode" in item:
    #                 # envs.reset()
    #                 # vecenvs reset automatically!
    #                 curr_reward = item["episode"]["r"]
    #                 # print(f"Episode reward: {curr_reward}")
    #     envs.close()
    # except Exception:
    #     print("Failed to run")
    #     envs.close()
