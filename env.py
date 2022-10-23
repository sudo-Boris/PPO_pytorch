import gym


### For Atari
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                # episode_trigger=lambda t: t % 100 == 0,
            )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_atari(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        ### Detail 1: When resetting the environment, randomly sample n no-ops between 1 and 30
        #   This is done to introduce some stochasticity into the environment
        env = NoopResetEnv(env, noop_max=30)
        ### Detail 2: Skip 4 frames
        #   This is done to reduce computation time
        #   In those skipped frames, the last action is repeated
        env = MaxAndSkipEnv(env, skip=4)
        ### Detail 3: For games where there are lives, end the episode when a life is lost
        #   But only reset the environment when the game is over (all lives are lost)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            ### Detail 4: For games where there is a fire button, press it after each reset
            #   Otherwise, the agent could be stuck in a stationary screen, unless it learns to press the fire button
            env = FireResetEnv(env)
        ### Detail 5: Clip rewards of the 4 skipped frames to be in [-1, 1]
        env = ClipRewardEnv(env)
        ### Detail 6: Image pre-processing/ transformations
        #   Resize the image to 84x84
        env = gym.wrappers.ResizeObservation(
            env, (84, 84)
        )  # Do this BEFORE the gray scale. There is a bug in the resize wrapper.
        #   Convert the image to grayscale
        env = gym.wrappers.GrayScaleObservation(env)
        ### Detail 7: Frame stacking
        #   Stack 4 frames together to create a single observation
        #   Intuitively, this can help the agent learn the velocity and direction of moving objects
        env = gym.wrappers.FrameStack(env, 4)
        ### This whole preprocessing transforms the input image of size (210, 160, 3) to (4, 84, 84)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_envs_atari(gym_id, seed, num_envs, capture_video, run_name):
    return gym.vector.SyncVectorEnv(
        [
            make_env_atari(gym_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]
    )


def make_envs(gym_id, seed, num_envs, capture_video, run_name):
    return gym.vector.SyncVectorEnv(
        [
            make_env(gym_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]
    )
