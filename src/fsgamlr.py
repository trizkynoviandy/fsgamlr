import numpy as np

class GeneticAlgorithm:

    def __init__(self, X, Y, max_features, population_size, n_generation, crossover_method):
        self.X = X
        self.Y = Y
        self.max_features = max_features
        self.population_size = population_size
        self.n_generation = n_generation
        self.n_genes = len(self.X.columns)
        self.population = []
        self.temp_fitness = []
        self.crossover_method = crossover_method
        self.offsprings = []
        self.fitness_offspring = []
        self.average_fitness = []
        self.best_fitness = []
        
    def generate_population(self):
        """
        Generate initial population
        """
        self.population = []
        for n in range(self.population_size):
            n_features = np.random.randint(1, self.max_features + 1)
            genes = ([1] * n_features) + ([0] * (self.n_genes - n_features))
            np.random.shuffle(genes)
            self.population.append(genes)
        return self.population