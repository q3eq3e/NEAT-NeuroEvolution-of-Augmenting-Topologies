import gymnasium as gym
import numpy as np
from enum import Enum
# from src.config import ENV_NAME
ENV_NAME = "Acrobot-v1"
# "CartPole-v1"
# "Pendulum-v1"
# "Acrobot-v1"


def get_params_num():
    env = gym.make(ENV_NAME)
    action_space = env.action_space
    observation_space = env.observation_space
    env.close()
    if isinstance(action_space, gym.spaces.Discrete):
        return observation_space.shape[0], action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        return observation_space.shape[0], action_space.shape[0]


class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2
    OTHER = 3


def run_env(model, render=False, seed=None):
    env = gym.make(ENV_NAME, render_mode="human")
    observation, _ = env.reset(seed=seed)

    action_type = ActionType.OTHER
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_type = ActionType.DISCRETE
    elif isinstance(env.action_space, gym.spaces.Box):
        action_type = ActionType.CONTINUOUS
    if action_type == ActionType.OTHER:
        raise ValueError("Game type not supported")

    terminated, truncated = False, False
    total_reward = 0

    while not terminated and not truncated:
        action = model.predict(observation)
        if action_type == ActionType.DISCRETE:
            action = np.argmax(action)
        elif action_type == ActionType.CONTINUOUS:
            action = action*(env.action_space.high-env.action_space.low)+env.action_space.low
            action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()

    env.close()
    return total_reward
