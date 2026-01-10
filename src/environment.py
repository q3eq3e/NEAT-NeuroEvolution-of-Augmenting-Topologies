import gymnasium as gym
import numpy as np
from enum import Enum


class Environment:
    def __init__(self, env_name, seed=None, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.seed = seed

    def __del__(self):
        self.env.close()

    class ActionType(Enum):
        DISCRETE = 1
        CONTINUOUS = 2
        OTHER = 3

    def get_params_num(self):
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        if isinstance(action_space, gym.spaces.Discrete):
            return observation_space.shape[0], action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            return observation_space.shape[0], action_space.shape[0]

    def run_env(self, model):
        observation, _ = self.env.reset(seed=self.seed)

        action_type = self.ActionType.OTHER
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_type = self.ActionType.DISCRETE
        elif isinstance(self.env.action_space, gym.spaces.Box):
            action_type = self.ActionType.CONTINUOUS
        if action_type == self.ActionType.OTHER:
            raise ValueError("Game type not supported")

        terminated, truncated = False, False
        total_reward = 0

        while not terminated and not truncated:
            action = model.predict(observation)
            if action_type == self.ActionType.DISCRETE:
                action = np.argmax(action)
            elif action_type == self.ActionType.CONTINUOUS:
                action = action*(self.env.action_space.high-self.env.action_space.low)+self.env.action_space.low
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward

        return total_reward
