import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class GeneticAlgorithm:
    def __init__(self, X, Y, max_features, population_size, n_generation):
        self.X = pd.DataFrame(X)
        self.Y = Y
        self.max_features = max_features
        self.population_size = population_size
        self.n_generation = n_generation
        self.n_genes = X.shape[1]
        self.population = []
        self.temp_fitness = []
        self.offsprings = []
        self.fitness_offspring = []
        self.average_fitness = []
        self.best_fitness = []
        
    def _generate_population(self):
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
        selected_features = [[] for _ in range(self.population_size)]
        for chromosome in range(self.population_size):
            for gene in range(self.n_genes):
                if self.population[chromosome][gene] == 1:
                    temp_selected = self.X.columns.values[gene]
                    selected_features[chromosome].append(temp_selected)
                    
        for selected in selected_features:
            features = self.X[selected]
            target = self.Y
            if len(selected) == 0 or len(selected) > self.max_features:
                fitness = np.float64(1000)
            else:
                regressor = LinearRegression().fit(features, target)
                y_pred = regressor.predict(features)
                fitness = mean_squared_error(target, y_pred, squared=False)
            self.temp_fitness.append(fitness)
        
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
            if len(features_offspring.columns) == 0 or len(features_offspring.columns) > self.max_features:
                fitness = np.float64(1000)
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
        if mutation_chance < 99:
            mutated_gen = random.randint(0, self.n_genes - 1)
            if self.offsprings[0][mutated_gen] == 1:
                self.offsprings[0][mutated_gen] = 0
            else:
                self.offsprings[0][mutated_gen] = 1
            if self.offsprings[1][mutated_gen] == 1:
                self.offsprings[1][mutated_gen] = 0
            else:
                self.offsprings[1][mutated_gen] = 1
        
    def _elitism(self):
        """
        The function takes in the fitness of the offsprings and compares it to the current population. If
        the fitness of the offspring is less than the current population, then the offspring replaces the
        current population
        :return: The population after the generation of offsprings and elitism.
        """
        counter = 0
        for fitness_offspring in self.fitness_offsprings:
            current_highest_fitness = max(self.temp_fitness)
            temp_index = self.temp_fitness.index(current_highest_fitness)
            if fitness_offspring < current_highest_fitness:
                self.population[temp_index] = self.offsprings[counter]
                self.temp_fitness[temp_index] = self.fitness_offsprings[counter]
            counter += 1
        avg = sum(self.temp_fitness) / len(self.population)
        self.average_fitness.append(avg)
        best_individual = min(self.temp_fitness)
        self.best_fitness.append(best_individual)

        return self.population
    
    def optimize(self, verbose):
        self._generate_population()
        for generation in range(self.n_generation):
            calc_fitness = self._calculate_fitness()
            cross = self._crossover()
            mutation = self._mutation()
            elit = self._elitism()
            
            if verbose == 1:
                print(f"Iteration {generation + 1} | Current Best: {self.best_fitness[generation]:.3f} | Average: {self.average_fitness[generation]:.3f}")
            elif verbose == 2:
                print(f"Iteration {generation + 1} | Current Best: {self.best_fitness[generation]:.3f} | Average: {self.average_fitness[generation]:.3f}")
                print('Offspring 1: {:.3f}'.format(self.fitness_offsprings[0]), '| Offspring 2: {:.3f}'.format(self.fitness_offsprings[1]))
        
        self.check_selected = []
        index = self.temp_fitness.index(min(self.temp_fitness))
        check_count = 0
        for item in self.population[index]:
            if item == 1:
                self.check_selected.append(self.X.columns.values[check_count])
            check_count += 1

        print('\nThe genetic algorithm has been run for {} iterations'.format(self.n_generation))
        print('The best chromosome: {} | RMSE: {}'.format(index, min(self.temp_fitness)))
        print('Average Population Fitness:', self.average_fitness[-1])
        print('\nSelected variables:', self.check_selected)
        
    def plot_result(self):
        plt.title('RMSE in Each Generation')
        plt.plot(self.best_fitness, label='Best Fitness')
        plt.plot(self.average_fitness, label='Average Fitness')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('RMSE')
        plt.show()