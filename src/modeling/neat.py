import random
from src.modeling.genome import Genome
from src.modeling.activation import sigmoid, identity


class NEAT:
    def __init__(
        self,
        input_size,
        output_size,
        act=sigmoid,
        weight_mutation_rate=0.8,
        add_node_rate=0.03,
        add_connection_rate=0.05,
        compatibility_threshold=3,
        c1=1,
        c2=1,
        c3=0.5,
        best_individuals_copied=0.2
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.weight_mutation_rate = weight_mutation_rate
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.compatibility_threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.best_individuals_copied = best_individuals_copied
        self._innovation_number = 0
        self.genomes = []
        self.species = [[self.genomes]]

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

    def mutate_weights(self, genome: Genome) -> None:
        for gene in genome.get_genes():
            if random.random() < self.weight_mutation_rate:
                gene.weight += random.uniform(-0.5, 0.5)

    def mutate(
        self,
        genome: Genome
    ) -> None:
        self.mutate_weights(genome)

        if random.random() < self.add_node_rate:
            innovation_number = self.get_new_innovation_number()
            self.get_new_innovation_number()
            self.mutate_add_node(genome, innovation_number)

        if random.random() < self.add_connection_rate:
            innovation_number = self.get_new_innovation_number()
            self.mutate_add_connection(genome, innovation_number)

    def delta(self, genome1, genome2):
        genes1 = {gene.innovation_number: gene for gene in genome1.get_genes()}
        genes2 = {gene.innovation_number: gene for gene in genome2.get_genes()}

        excess_border = min(max(genes1.keys()), max(genes2.keys()))
        E = 0
        D = 0
        W = 0
        intersections = 0
        N = max(len(genes1), len(genes2))
        all_innovations = set(genes1.keys()).union(set(genes2.keys()))

        for innovation in all_innovations:
            gene1 = genes1.get(innovation)
            gene2 = genes2.get(innovation)
            if gene1 and gene2:
                W += abs(gene1.weight - gene2.weight)
                intersections += 1
            elif innovation < excess_border:
                D += 1
            else:
                E += 1

        if intersections > 0:
            W /= intersections
        return (self.c1 * E / N) + (self.c2 * D / N) + self.c3 * W

    def speciate(self):
        new_species = [[] for _ in range(len(self.species))]
        representatives = [random.choice(s) for s in self.species if s]
        for genome in self.genomes:
            for i, representative in enumerate(representatives):
                if self.delta(genome, representative) < self.compatibility_threshold:
                    new_species[i].append(genome)
                    break
            else:
                new_species.append([genome])
                representatives.append(genome)
        self.species = [s for s in new_species if s]

    def determine_offspring(self):
        species_fitness = []
        global_fitness = 0

        for species in self.species:
            species_fitness.append(0)
            for genome in species:
                global_fitness += genome.fitness
                species_fitness[-1] += genome.fitness

        avg_fitness = global_fitness / len(self.genomes)
        if avg_fitness == 0:
            return [len(species) for species in self.species]

        kids_per_species = [int(sf / avg_fitness) for sf in species_fitness]
        rest = len(self.genomes) - sum(kids_per_species)
        lucky_species = random.sample(range(len(kids_per_species)), rest)
        for ls in lucky_species:
            kids_per_species[ls] += 1
        return kids_per_species

    def reproduce(self):
        kids_per_species = self.determine_offspring()
        offspring = []

        for i in range(len(self.species)):
            if kids_per_species[i] == 0:
                continue

            copied_individuals = 0
            if self.best_individuals_copied < 1:
                copied_individuals = int(self.best_individuals_copied * kids_per_species[i])
            else:
                copied_individuals = min(int(self.best_individuals_copied), kids_per_species[i])
            kids_per_species[i] -= copied_individuals

            if copied_individuals > 0:
                self.species[i].sort(key=lambda x: x.fitness, reverse=True)
                offspring.extend(self.species[i][:copied_individuals])

            for _ in range(kids_per_species[i]):
                parent1 = random.choice(self.species[i])
                parent2 = random.choice(self.species[i])
                child = self.crossover(parent1, parent2)
                offspring.append(child)

        self.genomes = offspring
