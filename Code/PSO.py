import numpy as np
import random
from continueous_eval_functions import sphere, rastrigin, rosenbrock

# -----------------------
# Initialization
# -----------------------
def init_particles(population, dimensions, vel_max, bounds):
    X = np.random.uniform(bounds[0], bounds[1], (population, dimensions))
    V = np.random.uniform(-vel_max, vel_max, (population, dimensions))
    return X, V

# -----------------------
# Evaluation
# -----------------------
def evaluate(pop, function: str):
    if function == 'sphere':
        fitness = np.array([sphere(i) for i in pop])
    elif function == 'rosenbrock':
        fitness = np.array([rosenbrock(i) for i in pop])
    elif function == 'rastrigin':
        fitness = np.array([rastrigin(i) for i in pop])
    else:
        raise ValueError("Not a valid continuous evaluation function type")
    return fitness

# -----------------------
# PSO Core
# -----------------------
def pso(hyperparameters: dict, function: str = 'sphere'):
    # Extract hyperparameters
    population = hyperparameters['POPULATION']
    generations = hyperparameters['GENERATIONS']
    dimensions = hyperparameters['DIMENSIONS']
    bounds = hyperparameters['BOUNDS']
    w = hyperparameters['W']
    c1 = hyperparameters['C1']
    c2 = hyperparameters['C2']
    vel_max = hyperparameters['VEL_MAX']

    X, V = init_particles(population, dimensions, vel_max, bounds)
    fitness = evaluate(X, function)

    # Initialize personal and global bests
    pbest = X.copy()
    pbest_val = fitness.copy()
    gbest_index = np.argmin(pbest_val)
    gbest = pbest[gbest_index].copy()
    gbest_val = pbest_val[gbest_index]

    best_hist = []

    for g in range(generations):
        r1, r2 = np.random.rand(population, dimensions), np.random.rand(population, dimensions)
        
        # Update velocity
        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X))
        V = np.clip(V, -vel_max, vel_max)

        # Update position
        X = np.clip(X + V, bounds[0], bounds[1])

        # Evaluate new positions
        fitness = evaluate(X, function)

        # Update personal bests
        improved = fitness < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = fitness[improved]

        # Update global best
        best_index = np.argmin(pbest_val)
        if pbest_val[best_index] < gbest_val:
            gbest = pbest[best_index].copy()
            gbest_val = pbest_val[best_index]

        best_hist.append(gbest_val)

    return gbest, gbest_val, best_hist
