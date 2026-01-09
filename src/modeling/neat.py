from src.modeling.genome import Genome
from src.modeling.activation import sigmoid, identity


class NEAT:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.genomes = []
