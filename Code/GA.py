import numpy as np
import math
import random 
from continueous_eval_functions import sphere, rastrigin, rosenbrock

def init_pop(population, dimensions, bounds):
    return np.random.uniform(bounds[0], bounds[1], (population, dimensions))

# Evaluation function
def evaluate(pop, function: str):
    if function == 'sphere':
        fitness = np.array([sphere(i) for i in pop])
    elif function == 'rosenbrock':
        fitness = np.array([rosenbrock(i) for i in pop])
    elif function == 'rastrigin':
        fitness = np.array([rastrigin(i) for i in pop])
    else: 
        raise ValueError("Not a valid continueous evaluation function type")
    return fitness

def tournament_select(pop, fitness, population, k = 3):
    i = np.random.randint(0, population, k)
    # Return the fitness minimizing individual among the k randomly selected individual
    return pop[i[np.argmin(fitness[i])]].copy()

def sbx_crossover(parent1, parent2, bounds,  eta = 10, crossover_chance = 0.5):
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if random.random() <= crossover_chance:
            # x1 is smaller val and x2 is larger val
            x1, x2 = sorted([parent1[i], parent2[i]])
            # Only crossover if difference bbetween the two is substantial
            if x2 - x1 > 1e-14:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u ))) ** (1 / eta + 1)
                    
                # Calculate and clip (bind) values by previously mentioned bounds
                child1[i] = np.clip(0.5 * ((1 + beta) * x1 + (1 - beta) * x2), bounds[0], bounds[1])
                child2[i] = np.clip(0.5 * ((1 - beta) * x1 + (1 + beta) * x2), bounds[0], bounds[1])
    return child1, child2

def mutate(child, mutation_rate, sigma, bounds):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            # Small mutation that is clipped
            child[i] = np.clip(child[i] + np.random.normal(0, sigma), bounds[0], bounds[1])
        return child


def ga(hyperparameters: dict, function: str = 'sphere'):
    # Extract hyperparameters
    population = hyperparameters['POPULATION']
    generations = hyperparameters['GENERATIONS']
    dimensions = hyperparameters['DIMENSIONS']
    bounds = hyperparameters['BOUNDS']
    crossover_rate = hyperparameters['CROSSOVER_RATE']
    mutation_rate = hyperparameters['MUTATION_RATE']
    elite_count = hyperparameters['ELITE_COUNT']
    sigma = hyperparameters['SIGMA']

    pop = init_pop(population, dimensions, bounds)
    fitness = evaluate(pop, function)
    best_hist = []
    for g in range(generations):
        new_pop = []
        elite_indices = np.argsort(fitness)[:elite_count]
        for index in elite_indices:
            new_pop.append(pop[index].copy())
        while len(new_pop) < population:
            parent1 = tournament_select(pop, fitness, population)
            parent2 = tournament_select(pop, fitness, population)
            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = sbx_crossover(parent1, parent2, bounds)
            else:
                c1, c2 = parent1.copy(), parent2.copy()
            # Mutation
            c1 = mutate(c1, mutation_rate, sigma, bounds)
            c2 = mutate(c2, mutation_rate, sigma, bounds)
            new_pop.append(c1)
            if len(new_pop) < population: 
                new_pop.append(c2)
        pop = np.array(new_pop)
        fitness = evaluate(pop, function)
        best_hist.append(np.min(fitness))
    # Return best particle, best value, and all history
    return pop[np.argmin(fitness)], np.min(fitness), best_hist
