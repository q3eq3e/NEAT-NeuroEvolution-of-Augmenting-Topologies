from src.modeling.neat import NEAT
from src.environment import Environment
from src.config import ENV_NAME

if __name__ == "__main__":
    env = Environment(ENV_NAME)
    neat = NEAT(*env.get_params_num())
    env_display = Environment(ENV_NAME, None, "human")
    neat.train(env.run_env)
    best = neat.get_best()
    print(best)
    env_display.run_env(best.predict)
