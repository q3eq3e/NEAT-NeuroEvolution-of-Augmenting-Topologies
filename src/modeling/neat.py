import random
from src.modeling.genome import Genome
from src.modeling.nn import NN
from src.modeling.activation import sigmoid
from tqdm import tqdm
from copy import deepcopy


class NEAT:
    def __init__(
        self,
        input_size,
        output_size,
    ):
        self.input_size = input_size
        self.output_size = output_size

    def get_new_innovation_number(self):
        self._innovation_number += 1
        return self._innovation_number - 1

    def initialize_population(self, size):
        self.genomes = []
        for _ in range(size):
            nn = NN(self.input_size, self.output_size, self.act)
            connections_nr = len(nn.connections)
            genome = Genome.create_from_nn(nn)
            self.genomes.append(genome)
        self._innovation_number = connections_nr
        self.species = [self.genomes]

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        genes1 = {gene.innovation_number: gene for gene in parent1.get_genes()}
        genes2 = {gene.innovation_number: gene for gene in parent2.get_genes()}
        more_fit_parent = parent1 if parent1.fitness > parent2.fitness else parent2
        more_fit_parent_genes = genes1 if parent1.fitness > parent2.fitness else genes2

        child_genes = []
        all_innovations = sorted(set(genes1.keys()).union(set(genes2.keys())))

        for innovation in all_innovations:
            gene1 = genes1.get(innovation)
            gene2 = genes2.get(innovation)
            more_fit_parent_gene = more_fit_parent_genes.get(innovation)

            if gene1 and gene2:
                chosen_gene = random.choice([gene1, gene2])
            else:
                chosen_gene = more_fit_parent_gene
                if chosen_gene is None:
                    continue
            child_genes.append(
                {
                    "weight": chosen_gene.weight,
                    "innovation_number": chosen_gene.innovation_number,
                    "enabled": chosen_gene.enabled,
                }
            )

        return Genome.create_from_parent(more_fit_parent, child_genes)

    def mutate_add_node(self, genome: Genome, innovation_number: int) -> None:
        connection = random.choice(genome.get_active_genes())
        genome.nn.add_node(connection, innovation_number)

    def mutate_add_connection(
        self, genome: Genome, innovation_number: int, mutation_range: float
    ):
        nodes = genome.nn.nodes

        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            from_node = random.choice(nodes)
            to_node = random.choice([n for n in nodes if not n.is_input_node()])
            if not genome.nn.conn_exists(from_node, to_node):
                genome.nn.add_connection(
                    from_node,
                    to_node,
                    innovation_number,
                    random.uniform(-mutation_range, mutation_range),
                )
                return True
            attempts += 1
        return False

    def mutate_weights(
        self, genome: Genome, weight_mutation_rate: float, mutation_range: float
    ) -> None:
        for gene in genome.get_active_genes():
            if random.random() < weight_mutation_rate:
                gene.set_weight(
                    gene.get_weight() + random.uniform(-mutation_range, mutation_range)
                )
        # bias mutation
        for node in genome.get_nn().nodes:
            if not node.is_input_node() and not node.is_output_node():
                if random.random() < weight_mutation_rate:
                    node.bias += random.uniform(-mutation_range, mutation_range)

    def _mutate(
        self,
        genome: Genome,
        weight_mutation_rate: float,
        mutation_range: float,
        add_node_rate: float,
        add_connection_rate: float,
    ) -> None:
        self.mutate_weights(genome, weight_mutation_rate, mutation_range)

        if random.random() < add_node_rate and len(genome.get_active_genes()) > 0:
            innovation_number = self.get_new_innovation_number()
            self.get_new_innovation_number()
            self.mutate_add_node(genome, innovation_number)

        if random.random() < add_connection_rate:
            innovation_number = self.get_new_innovation_number()
            if not self.mutate_add_connection(
                genome, innovation_number, mutation_range=0.5
            ):
                self._innovation_number -= 1  # rollback if no connection was added

    def mutate_population(
        self,
        weight_mutation_rate: float,
        mutation_range: float,
        add_node_rate: float,
        add_connection_rate: float,
    ):
        for genome in self.genomes:
            self._mutate(
                genome,
                weight_mutation_rate,
                mutation_range,
                add_node_rate,
                add_connection_rate,
            )

    def delta(self, genome1, genome2, c1: float, c2: float, c3: float):
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
            elif innovation <= excess_border:
                D += 1
            else:
                E += 1

        if intersections > 0:
            W /= intersections
        return (c1 * E / N) + (c2 * D / N) + c3 * W

    def speciate(
        self,
        c1: float,
        c2: float,
        c3: float,
        compatibility_threshold: float,
    ):
        new_species = [[] for _ in range(len(self.species))]
        representatives = [random.choice(s) for s in self.species if s]
        for genome in self.genomes:
            for i, representative in enumerate(representatives):
                if (
                    self.delta(genome, representative, c1, c2, c3)
                    < compatibility_threshold
                ):
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

        kids_per_species = [max(int(sf / avg_fitness), 0) for sf in species_fitness]
        rest = len(self.genomes) - sum(kids_per_species)
        lucky_species = random.sample(range(len(kids_per_species)), rest)
        for ls in lucky_species:
            kids_per_species[ls] += 1
        return kids_per_species

    def reproduce(self, best_individuals_copied: float):
        kids_per_species = self.determine_offspring()
        offspring = []

        for i in range(len(self.species)):
            if kids_per_species[i] == 0:
                continue

            copied_individuals = 0
            if best_individuals_copied < 1:
                # parameter treated as a fraction inside a species
                copied_individuals = int(best_individuals_copied * kids_per_species[i])
            else:
                # parameter treated as an absolute number inside a species
                copied_individuals = min(
                    int(best_individuals_copied), kids_per_species[i]
                )
            kids_per_species[i] -= copied_individuals

            if copied_individuals > 0:
                self.species[i].sort(key=lambda x: x.fitness, reverse=True)
                offspring.extend(deepcopy(self.species[i][:copied_individuals]))

            for _ in range(kids_per_species[i]):
                parent1 = random.choice(self.species[i])
                parent2 = random.choice(self.species[i])
                child = None
                if parent1 == parent2:
                    child = deepcopy(parent1)
                else:
                    child = self.crossover(parent1, parent2)
                offspring.append(child)

        self.genomes = offspring

    def train(
        self,
        evaluate,
        weight_mutation_rate=0.3,
        mutation_range=0.5,
        add_node_rate=0.01,
        add_connection_rate=0.01,
        compatibility_threshold=3,
        c1=0.5,
        c2=2,
        c3=2,
        best_individuals_copied=1,
        num_generations=50,
        population_size=100,
        act=sigmoid,
        verbose=False,
        callbacks=None,
    ):
        self.act = act
        self.initialize_population(population_size)
        self.best_found = self.genomes[0]
        callbacks = callbacks or []
        for epoch in tqdm(range(num_generations), disable=not verbose):
            for genome in self.genomes:
                genome.fitness = evaluate(genome)
            best_in_epoch = max(self.genomes, key=lambda x: x.fitness)
            if best_in_epoch.fitness > self.best_found.fitness:
                self.best_found = deepcopy(best_in_epoch)
            if verbose:
                print(
                    f"Generation {epoch} completed. Best fitness: {max(self.genomes, key=lambda x: x.fitness).fitness}"
                )
            self.speciate(c1, c2, c3, compatibility_threshold)
            self.reproduce(best_individuals_copied)
            self.mutate_population(
                weight_mutation_rate,
                mutation_range,
                add_node_rate,
                add_connection_rate,
            )
            if callbacks:
                stats = {
                    "fitness": self.get_best().fitness,
                    "species": self._get_species_sizes(),
                }

            for callback in callbacks:
                callback.log(epoch, stats)

    def get_best(self):
        return self.best_found

    def _get_species_sizes(self):
        return [len(s) for s in self.species]
