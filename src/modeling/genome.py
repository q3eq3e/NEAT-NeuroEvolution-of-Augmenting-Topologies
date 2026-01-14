from src.modeling.nn import NN
from src.modeling.node import Connection
from typing import List


class Genome:
    def __init__(self, genes: List[Connection] = None, nn=None, parent=None, info=None):

        # wymaga zmiany/poprawy
        if parent is not None:
            for gene in parent.get_genes():
                if not isinstance(gene, Connection):
                    raise ValueError("All genes must be instances of Conncection.")
            self.fitness: float = -1e6
            self.nn: NN = self._create_nn(parent, info)

        else:
            """Creates deepcopy of genes if NN is not provided."""
            for gene in genes:
                if not isinstance(gene, Connection):
                    raise ValueError("All genes must be instances of Conncection.")
            self.fitness: float = -1e6

            if nn is None:
                raise ValueError("you should not get there")
                self.nn: NN = self._create_nn(genes, info)
            else:
                if nn.connections != genes:
                    raise ValueError(
                        "Provided genes do not match the connections in the provided NN."
                    )
                self._genes = genes
                self.nn: NN = nn

    def create_from_nn(nn):
        """Creates Genome pointing at the same NN object."""
        return Genome(nn.connections, nn)

    def get_genes(self):
        return self.nn.connections

    def get_active_genes(self):
        return self.nn.active_connections()

    def get_nn(self):
        return self.nn

    def _create_nn(self, parent, info):
        nn = NN.create_from_parent(parent, info)
        self._genes = nn.connections
        return nn

    def __str__(self):
        res = "Genome:\n"
        for gene in self.get_genes():
            res += "   " + str(gene) + "\n"
        return res

    def predict(self, input):
        return self.nn.calculate_output(input)
