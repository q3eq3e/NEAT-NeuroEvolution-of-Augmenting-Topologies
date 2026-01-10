from numpy.random.mtrand import sample
from src.modeling.genome import Genome
from src.modeling.activation import sigmoid, identity
from random import choice


class NEAT:
    def __init__(self, input_size, output_size, act=sigmoid, c1=1, c2=1, c3=0.5, compatibility_threshold=3, best_individuals_copied=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.genomes = []
        self.species = [[self.genomes]]
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.compatibility_threshold = compatibility_threshold
        self.best_individuals_copied = best_individuals_copied

    def delta(self, genome1, genome2):
        gens1 = [gen.innovation_number for gen in genome1.nn.connections]
        gens2 = [gen.innovation_number for gen in genome2.nn.connections]
        max_innovation1 = max(gens1)
        max_innovation2 = max(gens2)
        excess_border = min(max_innovation1, max_innovation2)
        E = 0
        D = 0
        W = 0
        N = max(len(gens1), len(gens2))
        for gen in gens1:
            if gen > excess_border:
                E += 1
            elif gen in gens2:
                W += abs(genome1.nn.connections[gen].weight - genome2.nn.connections[gen].weight)
            else:
                D += 1
        W = W / (len(gens1) - E - D)
        for gen in gens2:
            if gen > excess_border:
                E += 1
            elif gen not in gens1:
                D += 1
        return (self.c1 * E / N) + (self.c2 * D / N) + self.c3 * W

    def speciate(self):
        new_species = [[] for _ in range(len(self.species))]
        representatives = [choice(s) for s in self.species if s]
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
        if not global_fitness:
            return
        avg_fitness = global_fitness / len(self.genomes)
        kids_per_species = [int(sf / avg_fitness) for sf in species_fitness]
        rest = len(self.genomes) - sum(kids_per_species)
        lucky_species = sample(range(len(kids_per_species)), rest)
        for ls in lucky_species:
            kids_per_species[ls] += 1
        return kids_per_species

    def reproduce(self):
        kids_per_species = self.determine_offspring()
        offspring = []
        for i, species in enumerate(self.species):
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
                parent1 = choice(self.species[i])
                parent2 = choice(self.species[i])
                child = self.crossover(parent1, parent2)
                offspring.append(child)
        self.genomes = offspring
