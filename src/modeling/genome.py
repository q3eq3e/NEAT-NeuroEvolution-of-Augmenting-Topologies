from src.modeling.nn import NN
from src.modeling.node import Connection
from typing import List


class Genome:
    def __init__(self, genes: List[Connection], nn):
        for gene in genes:
            if not isinstance(gene, Connection):
                raise ValueError("All genes must be instances of Conncection.")

        if nn.connections != genes:
            raise ValueError(
                "Provided genes do not match the connections in the provided NN."
            )
        self.fitness: float = -1e6
        self.nn: NN = nn

    def create_from_nn(nn):
        """Creates Genome pointing at the same NN object."""
        return Genome(nn.connections, nn)

    def _create_nn(parent, info):
        return NN.create_from_parent(parent, info)

    def create_from_parent(parent, info_genes):
        for gene in parent.get_genes():
            if not isinstance(gene, Connection):
                raise ValueError("All genes must be instances of Conncection.")
        return Genome.create_from_nn(Genome._create_nn(parent, info_genes))

    def get_genes(self):
        return self.nn.connections

    def get_active_genes(self):
        return self.nn.active_connections()

    def get_nn(self):
        return self.nn

    def __str__(self):
        res = "Genome:\n"
        for gene in self.get_genes():
            res += "   " + str(gene) + "\n"
        return res

    def predict(self, input):
        return self.nn.calculate_output(input)
