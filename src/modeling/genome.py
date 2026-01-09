from src.modeling.nn import NN
from src.modeling.node import Connection
from typing import List


class Genome:
    def __init__(self, genes, nn=None):
        for gene in genes:
            if not isinstance(gene, Connection):
                raise ValueError("All genes must be instances of Conncection.")
        self._genes: List[Connection] = genes
        # self.fitness: float = 0.0
        # self.adjusted_fitness: float = 0.0

        if nn is None:
            self.nn: NN = self._create_nn()
        else:
            self.nn: NN = nn

    def create_from_nn(nn):
        return Genome(nn.connections, nn)

    def get_genes(self):
        return self.nn.connections

    def get_nn(self):
        return self.nn

    def _create_nn(self):
        return NN.create_from_genome(self._genes)

    def __str__(self):
        res = ""
        for gene in self.get_genes():
            res += str(gene) + "\n"
        return res
