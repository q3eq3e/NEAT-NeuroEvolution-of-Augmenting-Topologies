import os
import sys
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/..")

from src.modeling.neat import NEAT  # noqa:E402
from src.modeling.activation import sigmoid, tanh  # noqa: E402, F401
from src.environment import Environment  # noqa: E402
from src.environment import store_episode_as_gif  # noqa: E402
from src.config import ENV_NAME  # noqa: E402


if __name__ == "__main__":
    env = Environment(ENV_NAME, seed=2137)
    neat = NEAT(*env.get_params_num())
    neat.train(
        env.run_env,
        weight_mutation_rate=0.3,
        mutation_range=0.5,
        add_node_rate=0.01,
        add_connection_rate=0.01,
        compatibility_threshold=3,
        c1=0.5,
        c2=2,
        c3=2,
        best_individuals_copied=1,
        num_generations=2,
        population_size=1000,
        act=sigmoid,
        verbose=True,
    )
    best = neat.get_best()
    print(best)
    print(best.nn)
    print(best.fitness)
    with open("species.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(neat.get_species_size_overtime())
    reward, frames = env.run_env(best, store_gif=True)
    print("reward:", reward)
    store_episode_as_gif(frames, filename="gif.gif")
