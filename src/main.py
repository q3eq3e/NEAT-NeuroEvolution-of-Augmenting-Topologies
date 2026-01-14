from src.modeling.neat import NEAT
from src.environment import Environment
from src.environment import store_episode_as_gif
from src.config import ENV_NAME
from src.logger import FitnessLogger
import pickle
import datetime
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEAT")

    parser.add_argument(
        "model_file",
        nargs="?",
        help="Path to .pkl file",
    )

    args = parser.parse_args()

    env = Environment(ENV_NAME)

    if args.model_file:
        with open(args.model_file, "rb") as f:
            best = pickle.load(f)

    else:

        neat = NEAT(*env.get_params_num())
        fit_log = FitnessLogger()
        neat.train(env.run_env, callbacks=[fit_log])
        fit_log.save_chart_data("fitness_chart.png")
        best = neat.get_best()

        with open(
            f"{ENV_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb"
        ) as f:
            pickle.dump(best, f)

    print(best)
    print(best.nn)
    print(best.fitness)
    _, frames = env.run_env(best, store_gif=True)
    store_episode_as_gif(frames, filename="gif.gif")
