import random
from src.modeling.genome import Genome
from src.modeling.activation import sigmoid, identity


class NEAT:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.genomes = []
        self._innovation_number = 0

    def get_new_innovation_number(self):
        self._innovation_number += 1
        return self._innovation_number - 1

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        genes1 = {gene.innovation_number: gene for gene in parent1.get_genes()}
        genes2 = {gene.innovation_number: gene for gene in parent2.get_genes()}

        child_genes = []
        all_innovations = set(genes1.keys()).union(set(genes2.keys()))

        for innovation in all_innovations:
            gene1 = genes1.get(innovation)
            gene2 = genes2.get(innovation)

            if gene1 and gene2:
                chosen_gene = random.choice([gene1, gene2])
            elif gene1:
                chosen_gene = gene1
            else:
                chosen_gene = gene2

            child_genes.append(chosen_gene)

        return Genome(child_genes)

    def mutate_add_node(self, genome: Genome, innovation_number: int) -> None:
        connection = random.choice(genome.get_genes())
        while not connection.enabled:
            connection = random.choice(genome.get_genes())

        genome.nn.add_node(connection, innovation_number)

    def mutate_add_connection(self, genome: Genome, innovation_number: int) -> None:
        nodes = genome.nn.nodes

        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            from_node = random.choice(nodes)
            to_node = random.choice(nodes)
            if from_node != to_node and not genome.nn.conn_exists(from_node, to_node):
                genome.nn.add_connection(
                    from_node, to_node, innovation_number, random.uniform(-1.0, 1.0)
                )
                return
            attempts += 1

    def mutate_weights(self, genome: Genome, mutation_rate=0.8) -> None:
        for gene in genome.get_genes():
            if random.random() < mutation_rate:
                gene.weight += random.uniform(-0.5, 0.5)

    def mutate(
        self,
        genome: Genome,
        weight_mutation_rate=0.8,
        add_node_rate=0.03,
        add_connection_rate=0.05,
    ) -> None:
        if random.random() < weight_mutation_rate:
            self.mutate_weights(genome)

        if random.random() < add_node_rate:
            innovation_number = self.get_new_innovation_number()
            self.mutate_add_node(genome, innovation_number)

        if random.random() < add_connection_rate:
            innovation_number = self.get_new_innovation_number()
            self.get_new_innovation_number()
            self.mutate_add_connection(genome, innovation_number)
