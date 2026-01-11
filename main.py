from src.modeling.neat import NEAT
from src.environment import Environment
from src.environment import store_episode_as_gif
from src.config import ENV_NAME


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
