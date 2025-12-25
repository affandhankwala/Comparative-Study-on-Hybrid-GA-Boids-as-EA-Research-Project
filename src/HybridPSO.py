import numpy as np
import random
from continueous_eval_functions import sphere, rastrigin, rosenbrock

def init_particles(population, dimensions, vel_max, bounds):
    X = np.random.uniform(bounds[0], bounds[1], (population, dimensions))
    V = np.random.uniform(-vel_max, vel_max, (population, dimensions))
    return X, V

def evaluate(pop, function: str):
    if function == 'sphere':
        return np.array([sphere(x) for x in pop])
    elif function == 'rastrigin':
        return np.array([rastrigin(x) for x in pop])
    elif function == 'rosenbrock':
        return np.array([rosenbrock(x) for x in pop])
    else:
        raise ValueError("Invalid continuous function")

def hybrid_pso(hyperparameters: dict, function='sphere'):
    # Extract hyperparameters
    population = hyperparameters['POPULATION']
    generations = hyperparameters['GENERATIONS']
    dimensions = hyperparameters['DIMENSIONS']
    bounds = hyperparameters['BOUNDS']
    vel_max = hyperparameters['VEL_MAX']
    w = hyperparameters['W']
    c1 = hyperparameters['C1']
    c2 = hyperparameters['C2']
    mutation_rate = hyperparameters['MUTATION_RATE']
    sigma = hyperparameters['SIGMA']
    stagnation_limit = hyperparameters['STAGNATION_LIMIT']

    X, V = init_particles(population, dimensions, vel_max, bounds)
    fitness = evaluate(X, function)
    pbest = X.copy()
    pbest_val = fitness.copy()

    gbest = pbest[np.argmin(pbest_val)].copy()
    gbest_val = np.min(pbest_val)
    best_hist = []

    no_improve = 0

    for g in range(generations):
        r1, r2 = np.random.rand(population, dimensions), np.random.rand(population, dimensions)
        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X))
        V = np.clip(V, -vel_max, vel_max)
        X = np.clip(X + V, bounds[0], bounds[1])

        # Mutation phase if swarm stagnates
        if no_improve >= stagnation_limit:
            for i in range(population):
                if random.random() < mutation_rate:
                    X[i] += np.random.normal(0, sigma, dimensions)
                    X[i] = np.clip(X[i], bounds[0], bounds[1])
            no_improve = 0  # reset stagnation counter

        # Evaluate
        fitness = evaluate(X, function)
        improved = fitness < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = fitness[improved]

        best_index = np.argmin(pbest_val)
        if pbest_val[best_index] < gbest_val:
            gbest_val = pbest_val[best_index]
            gbest = pbest[best_index].copy()
            no_improve = 0
        else:
            no_improve += 1

        best_hist.append(gbest_val)
        
    return gbest, gbest_val, best_hist
