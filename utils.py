import argparse
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default="PPO",
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="CartPole-v1",
        # default="MountainCar-v0",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,  # 25000,
        help="total timesteps of the experiments. Num_updates = total_timesteps // batch_size",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="PPO_Projects",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of parallel gym environments"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="number of steps to run in each environment per policy rollout.\
            Rollout Data = num_envs * num_steps",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="gae lambda parameter"
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="number of mini batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="number of epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="ppo surrogate cliping coefficient (epsilon in the paper)",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles wheter or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy. Slow down minimization of entropy. \
            It helps exploration by limiting the premature convergence to suboptimal policy.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="value function loss coefficient in the optimization objective",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum value for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,  # 0.015,
        help="the target KL divergence threshold. If not None,\
            stop training prematurely when the KL divergence between old and new policy\
            is smaller than this value",
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    assert (
        args.batch_size % args.num_minibatches == 0
    ), "Batch size should be divisible by number of mini batches"
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


def init_tb_writer(args, run_name):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    return writer


if __name__ == "__main__":
    args = parse_args()
    print(args)
