import gym


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


def make_envs(gym_id, seed, num_envs, capture_video, run_name):
    return gym.vector.SyncVectorEnv(
        [
            make_env(gym_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]
    )
