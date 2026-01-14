from src.modeling.nn import NN
from src.modeling.node import Connection
from typing import List


class Genome:
    def __init__(self, nn: NN) -> None:
        """Creates Genome pointing at the same NN object."""
        self.fitness: float = -1e6
        self.nn: NN = nn

    @staticmethod
    def _create_nn(parent: "Genome", info: dict) -> NN:
        return NN.create_from_parent(parent, info)

    @staticmethod
    def create_from_parent(parent: "Genome", info_genes: dict) -> "Genome":
        for gene in parent.get_genes():
            if not isinstance(gene, Connection):
                raise ValueError("All genes must be instances of Conncection.")
        return Genome(Genome._create_nn(parent, info_genes))

    def get_genes(self) -> List[Connection]:
        return self.nn.connections

    def get_active_genes(self) -> List[Connection]:
        return self.nn.active_connections()

    def get_nn(self) -> NN:
        return self.nn

    def __str__(self) -> str:
        res = "Genome:\n"
        for gene in self.get_genes():
            res += "   " + str(gene) + "\n"
        return res

    def predict(self, input: list[float]) -> list[float]:
        return self.nn.calculate_output(input)
