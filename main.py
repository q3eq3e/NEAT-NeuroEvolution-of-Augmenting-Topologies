from src.modeling.neat import NEAT
from src.modeling.nn import NN
from src.environment import Environment
from src.modeling.genome import Genome
from src.environment import store_episode_as_gif
from src.config import ENV_NAME
import random


#                                                   !!!!!!!!!! zmien plik
if __name__ == "__main__":
    env = Environment(ENV_NAME)
    neat = NEAT(*env.get_params_num())
    neat.train(env.run_env)
    best = neat.get_best()
    print(best)
    print(best.nn)
    print(best.fitness)
    _, frames = env.run_env(best, store_gif=True)
    store_episode_as_gif(frames, filename="gif.gif")

    # nn1 = NN(4, 1)
    # nn1.add_node(nn1.active_connections()[3], innovation=4)
    # nn1.add_connection(nn1.nodes[3], nn1.nodes[5], 8, weight=0.5)
    # print(nn1)
    # parent1 = Genome.create_from_nn(nn1)
    # parent1.fitness = 10.0

    # nn2 = NN(4, 1)
    # nn2.add_node(nn2.active_connections()[0], innovation=6)
    # nn2.add_connection(nn2.nodes[1], nn2.nodes[4], 9, weight=0.8)
    # print(nn2)
    # parent2 = Genome.create_from_nn(nn2)
    # parent2.fitness = 0.0

    # child = crossover(parent1, parent2)
    # print(child)
    # print(child.get_nn())
