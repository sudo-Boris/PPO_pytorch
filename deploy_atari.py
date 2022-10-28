import torch
import gym
import agent
from env import make_deploy_env_atari


def deploy(
    model_path: str = "trained_models/BreakoutNoFrameskip-v0_PPO_1_Oct25_12-19-52.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gym_id = model_path.split("/")[1].split("_")[0]

    def make_env():
        def thunk():
            return make_deploy_env_atari(
                gym_id=gym_id,
            )

        return thunk()

    def make_envs():
        return gym.vector.SyncVectorEnv([make_env])

    envs = make_envs()

    # Define model
    model = agent.AtariAgent(envs=envs)
    # Load model
    model.load_state_dict(torch.load(model_path))
    # Set model to eval mode
    model.eval()

    # Start run
    done = False
    obs = envs.reset()
    while not done:
        action, _, _, _ = model.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, done, _ = envs.step(action.cpu().numpy())

    envs.close()


def main():
    model = "BreakoutNoFrameskip-v0_PPO_1_Oct25_12-19-52.pt"
    deploy(model_path=f"trained_models/{model}")


if __name__ == "__main__":
    main()
