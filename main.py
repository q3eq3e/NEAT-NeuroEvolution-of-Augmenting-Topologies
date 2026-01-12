from src.modeling.neat import NEAT
from src.modeling.nn import NN
from src.environment import Environment
from src.modeling.genome import Genome
from src.environment import store_episode_as_gif
from src.config import ENV_NAME
import random
import csv


#                                                   !!!!!!!!!! zmien plik
if __name__ == "__main__":
    env = Environment(ENV_NAME, seed=2137)
    neat = NEAT(*env.get_params_num())
    fine_tune = False
    if fine_tune:
        i = 1
        overall_best = None
        best_fit = -1e6
        best_i = 1
        for population_size in [30]:
            # iterations = int(500 / population_size)
            iterations = 20
            for weight_mut in [0.3, 0.7]:
                mut_range = 0.15 / weight_mut
                # for mut_range in [0.2, 0.5]:
                for node_mut in [0.01, 0.1]:
                    for conn_mut in [0.01, 0.1]:
                        for best_copied in [0.1, 1, 3]:
                            for c1 in [0.5, 1, 2]:
                                for c2 in [0.5, 1, 2]:
                                    for c3 in [0.5, 1, 2]:
                                        threshold = 2
                                        if i == 198:
                                            # 0.3 0.5 0.1 0.01 1 0.5 2 2
                                            print(
                                                weight_mut,
                                                mut_range,
                                                node_mut,
                                                conn_mut,
                                                best_copied,
                                                c1,
                                                c2,
                                                c3,
                                                "\n\n",
                                            )
                                            # for threshold in [1, 2, 3, 5]:
                                        print(
                                            f"approach {i*100.0/648}% - nr {i} out of 648"
                                        )
                                        neat.train(
                                            env.run_env,
                                            weight_mut,
                                            mut_range,
                                            node_mut,
                                            conn_mut,
                                            threshold,
                                            c1,
                                            c2,
                                            c3,
                                            best_copied,
                                            iterations,
                                            population_size,
                                        )
                                        best = neat.get_best()
                                        print(best.fitness)
                                        if best.fitness > best_fit:
                                            overall_best = best
                                            best_fit = best.fitness
                                            best_i = i
                                        i += 1

        # neat.train(env.run_env)
        # best = neat.get_best()
        # print(best)
        # print(best.nn)
        # print(best.fitness)

        print(overall_best)
        print("best i:", best_i)
        print(overall_best.nn)
        print(overall_best.fitness)
        _, frames = env.run_env(overall_best, store_gif=True)
        store_episode_as_gif(frames, filename="gif.gif")
    else:
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
            num_generations=200,
            population_size=2000,
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
