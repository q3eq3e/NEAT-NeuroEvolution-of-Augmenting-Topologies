import gymnasium as gym
import numpy as np
from enum import Enum
from matplotlib import animation
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, env_name, seed=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
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
            return observation_space.shape[0], int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            return observation_space.shape[0], action_space.shape[0]

    def run_env(self, model, store_gif=False):
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
        frames = []

        while not terminated and not truncated:
            action = model.predict(observation)
            if action_type == self.ActionType.DISCRETE:
                action = np.argmax(action)
            # elif action_type == self.ActionType.CONTINUOUS:
            #     action = clip(action, self.env.action_space.low, self.env.action_space.high)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if store_gif:
                frames.append(self.env.render())

        # if store_gif:
        #     store_episode_as_gif(frames, gif_path, gif_filename)

        if store_gif:
            return total_reward, frames

        return total_reward


def store_episode_as_gif(frames, path="./", filename="animation.gif"):
    """Store episode as gif animation"""
    fps = 30  # Set framew per seconds
    dpi = 100  # Set dots per inch
    interval = 5  # Interval between frames (in ms)

    # Fix frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    # Generate animation
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval
    )

    # Save output as gif
    anim.save(path + filename, writer="imagemagick", fps=fps)
