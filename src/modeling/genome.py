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

    def create_from_nn(nn):
        return Genome(nn.connections, nn)

    def get_genes(self):
        return self.nn.connections

    def get_active_genes(self):
        return self.nn.active_connections()

    def get_nn(self):
        return self.nn

    def _create_nn(self):
        input_mid_nodes = set()
        output_mid_nodes = set()
        for conn in self.genes:
            output_mid_nodes.add(conn.get_source_node())
            input_mid_nodes.add(conn.get_target_node())
        input_nodes = self.nodes not in output_mid_nodes
        output_nodes = self.nodes not in input_mid_nodes
        nn = NN(len(input_nodes), len(output_nodes))
        # requires cahnge
        for conn in self.genes:
            if (
                conn.get_source_node() in nn.nodes
                and conn.get_target_node() in nn.nodes
            ):
                nn.add_connection(
                    conn.get_source_node(),
                    conn.get_target_node(),
                    conn.innovation_number,
                    conn.get_weight(),
                    conn.enabled,
                )
            elif conn.get_source_node() in nn.nodes:
                to_node = next(conn).get_target_node()
                nn.add_node(
                    nn.get_connection(conn.get_source_node(), to_node),
                    conn.innovation_number,
                )
            # wagi

        return nn
