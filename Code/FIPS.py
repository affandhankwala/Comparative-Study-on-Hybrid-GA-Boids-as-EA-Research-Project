import numpy as np
from continueous_eval_functions import sphere, rastrigin, rosenbrock

# -----------------------
# Initialization
# -----------------------
def init_particles(bounds, population, dimensions, vel_max):
    X = np.random.uniform(bounds[0], bounds[1], (population, dimensions))
    V = np.random.uniform(-vel_max, vel_max, (population, dimensions))
    return X, V

# -----------------------
# Evaluation
# -----------------------
def evaluate(pop, function: str):
    if function == 'sphere':
        return np.array([sphere(x) for x in pop])
    elif function == 'rastrigin':
        return np.array([rastrigin(x) for x in pop])
    elif function == 'rosenbrock':
        return np.array([rosenbrock(x) for x in pop])
    else:
        raise ValueError("Invalid continuous function")

# -----------------------
# FIPS implementation
# -----------------------
def fips(hyperparameters: dict, function='sphere'):
    # Hyperparameter assignment
    population = hyperparameters['POPULATION']
    generations = hyperparameters['GENERATIONS']
    dimensions = hyperparameters['DIMENSIONS']
    bounds = hyperparameters['BOUNDS']
    vel_max = hyperparameters['VEL_MAX']
    w = hyperparameters['W']
    c = hyperparameters['C']
    k = hyperparameters['K']

    X, V = init_particles(bounds, population, dimensions, vel_max)
    fitness = evaluate(X, function)
    pbest = X.copy()
    pbest_val = fitness.copy()
    best_hist = []

    for g in range(generations):
        # Precompute best neighbors for each particle
        for i in range(population):
            neighbors = np.random.choice(population, k, replace=False)
            influence = np.zeros(dimensions)
            for j in neighbors:
                r = np.random.rand(dimensions)
                influence += r * (pbest[j] - X[i])
            V[i] = w * V[i] + (c / k) * influence
            V[i] = np.clip(V[i], -vel_max, vel_max)
            X[i] = np.clip(X[i] + V[i], bounds[0], bounds[1])

        # Evaluate
        fitness = evaluate(X, function)
        improved = fitness < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = fitness[improved]

        # Track global best
        gbest_val = np.min(pbest_val)
        best_hist.append(gbest_val)

    return pbest[np.argmin(pbest_val)], np.min(pbest_val), best_hist
