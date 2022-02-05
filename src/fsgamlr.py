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
    
    def _crossover(self):
        """
        The function is responsible for selecting two parents from the population, and then generating
        two offsprings by performing crossover on the selected parents.
        :return: The fitness of the offsprings.
        """
        parent_1 = self._selection()
        parent_2 = self._selection()  
        self.offsprings = []

        crossover_point = int(self.n_genes / 2)

        parent_1a = parent_1[0:crossover_point]
        parent_2a = parent_2[crossover_point:]
        parent_1b = parent_2[0:crossover_point]
        parent_2b = parent_1[crossover_point:]

        temp_offspring_1 = parent_1a + parent_2a
        temp_offspring_2 = parent_1b + parent_2b
        self.offsprings.append(temp_offspring_1)
        self.offsprings.append(temp_offspring_2)

        offspring_selected_features = []

        for offspring in self.offsprings:
            temp_selected = []
            for genes in range(self.n_genes):
                if offspring[genes] == 1:
                    temp_selected_offspring = self.X.columns.values[genes]
                    temp_selected.append(temp_selected_offspring)
            offspring_selected_features.append(temp_selected)
        
        self.fitness_offsprings = []

        for features in offspring_selected_features:
            features_offspring = self.X[features]
            target_offspring = self.Y
            if len(features_offspring.columns) == 0:
                self.fitness_offsprings.append(np.float64(1000))
            elif len(features_offspring.columns) > self.max_features:
                self.fitness_offsprings.append(np.float64(1000))
            else:
                regressor = LinearRegression().fit(features_offspring, target_offspring)
                y_pred = regressor.predict(features_offspring)
                fitness = mean_squared_error(target_offspring, y_pred, squared=False)
                self.fitness_offsprings.append(fitness)

        return self.fitness_offsprings
    
    def _mutation(self):
        """
        - The function is used to mutate the offspring.
            - The function is called when the mutation chance is 5.
            - The function is used to mutate the offspring.
            - The function is called when the mutation chance is 5.
        """
        mutation_chance = random.randint(1, 100)
        if mutation_chance == 5:
            print('terjadi Mutasi')
            mutated_gen = random.randint(0, self.n_genes)
            if self.offsprings[0][mutated_gen] == 1:
                self.offsprings[0][mutated_gen] == 0
            else:
                self.offsprings[0][mutated_gen] == 1
            if self.offsprings[1][mutated_gen] == 1:
                self.offsprings[1][mutated_gen] == 0
            else:
                self.offsprings[1][mutated_gen] == 1
        else:
            pass
        
    def _elitism(self):
        """
        The function takes in the fitness of the offsprings and compares it to the fitness of the parents.
        If the fitness of the offspring is less than the fitness of the parents, then the offspring replaces
        the parent
        :return: The population after the generation of offsprings and elitism.
        """
        counter = 0
        for fitness_offspring in self.fitness_offsprings:
            current_highest_fitness = [max(self.temp_fitness)]
            temp_index = self.temp_fitness.index(max(self.temp_fitness))
            if fitness_offspring < current_highest_fitness:
                self.population[temp_index] = self.offsprings[counter]
                self.temp_fitness[temp_index] = self.fitness_offsprings[counter]
            else:
                pass
            counter +=1
        avg = sum(self.temp_fitness) / len(self.population)
        self.average_fitness.append(avg)
        best_individual = min(self.temp_fitness)
        self.best_fitness.append(best_individual)

        return self.population