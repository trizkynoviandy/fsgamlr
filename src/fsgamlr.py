import random

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        Generate a population of random candidate solutions.
        
        :param self: This is the object that is being called
        :return: The population.
        """
        for n in range(self.population_size):
            n_features = np.random.randint(1, self.max_features + 1)
            genes = ([1] * n_features) + ([0] * (self.n_genes - n_features))
            np.random.shuffle(genes)
            self.population.append(genes)
        return self.population
    
    def _calculate_fitness(self):
        """
        Calculate the fitness of each chromosome in the population.
        
        :param self: This is the object that is being called
        :return: The fitness of each chromosome.
        """
        self.temp_fitness = []
        for generation in range(self.n_generation):
            selected_features = [[] for _ in range(self.population_size)]

            for chromosome in range(self.population_size):
                for gene in range(self.n_genes):
                    if self.population[chromosome][gene] == 1:
                        temp_selected = self.X.columns.values[gene]
                        selected_features[chromosome].append(temp_selected)

        count = 0
        for selected in selected_features:
            features = self.X[selected]
            target = self.Y

            if len(selected_features[count]) == 0:
                self.temp_fitness.append(fitness**2)
            elif len(selected_features[count]) > self.max_features:
                self.temp_fitness.append(fitness**2)
            else:
                regressor = LinearRegression().fit(features,target)
                y_pred = regressor.predict(features)
                fitness = mean_squared_error(target, y_pred, squared=False)
                self.temp_fitness.append(fitness)
            count +=1

        return self.temp_fitness
    
    def _selection(self):
        """
        Select a random number between 0 and the sum of the fitness values of the population. 
        Then, iterate through the population and add the fitness value of each individual to a roulette
        wheel. If the random number is between the current value of the roulette wheel and the previous 
        value of the roulette wheel, then the parent is the individual at the current index.
        
        :param self: refers to the object that is calling the method
        :return: The parent.
        """
        temp_parent = []
        temp_roulette = []
        roulette_wheels = sum(self.temp_fitness)

        for item in range(self.population_size):
            if item == 0:
                temp_roulette.append(self.temp_fitness[0])
            else:
                temp_roulette.append(temp_roulette[item - 1] + self.temp_fitness[item])

        selection = random.uniform(0, float(roulette_wheels))

        for counter in range(self.population_size -1):
            if selection <= temp_roulette[0]:
                temp_parent = self.population[0]
                break
            elif selection >= temp_roulette[counter] and selection <= temp_roulette[counter + 1]:
                temp_parent = self.population[counter]
                break

        return temp_parent