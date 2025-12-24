import numpy as np
import random
from continueous_eval_functions import sphere, rastrigin, rosenbrock

# Shared hyperparameters

def init_population(population, dimensions, vel_max, bounds):
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


def find_neighbors(X, i, neighbor_radius):
    distances = np.linalg.norm(X - X[i], axis=1)
    radius = neighbor_radius * np.mean(distances)
    neighbors = np.where(distances < radius)[0]
    neighbors = neighbors[neighbors != i]
    return neighbors if len(neighbors) > 0 else [i]


def tournament_select(pop, fitness, population, k=3):
    i = np.random.randint(0, population, k)
    return pop[i[np.argmin(fitness[i])]].copy()


def crossover(p1, p2, crossover_rate, dimensions):
    if random.random() < crossover_rate:
        alpha = np.random.uniform(0, 1, dimensions)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2
    return p1.copy(), p2.copy()


def mutate(child, dimensions, mutation_rate, sigma, bounds):
    for i in range(dimensions):
        if random.random() < mutation_rate:
            child[i] += np.random.normal(0, sigma)
    return np.clip(child, bounds[0], bounds[1])


def hybrid_ga_boids(hyperparameters: dict, function='sphere'):
    # Extract hyperparameters
    population = hyperparameters['POPULATION']
    generations = hyperparameters['GENERATIONS']
    dimensions = hyperparameters['DIMENSIONS']
    bounds = hyperparameters['BOUNDS']
    w = hyperparameters['W']                # Inertia weight
    alignment_weight = hyperparameters['ALIGNMENT_WEIGHT']
    cohesion_weight = hyperparameters['COHESION_WEIGHT']
    separation_weight = hyperparameters['SEPARATION_WEIGHT']
    global_best_attraction = hyperparameters['GLOBAL_BEST_ATTRACTION']
    vel_max = hyperparameters['VEL_MAX']
    neighbor_radius = hyperparameters['NEIGHBOR_RADIUS']
    crossover_rate = hyperparameters['CROSSOVER_RATE']
    mutation_rate = hyperparameters['MUTATION_RATE']
    sigma = hyperparameters['SIGMA']
    elite_count = hyperparameters['ELITE_COUNT']

    X, V = init_population(population, dimensions, vel_max, bounds)
    fitness = evaluate(X, function)
    best_hist = []

    gbest_index = np.argmin(fitness)
    gbest = X[gbest_index].copy()
    gbest_val = fitness[gbest_index]

    for g in range(generations):
        # ----- Boids phase -----
        new_V = np.zeros_like(V)
        for i in range(population):
            neighbors = find_neighbors(X, i, neighbor_radius)
            weights = 1 / (fitness[neighbors] + 1e-9)
            weights /= np.sum(weights)
            # Bias motion towards fitter neighbors
            alignment = np.average(V[neighbors], axis=0, weights=weights) - V[i]
            cohesion  = np.average(X[neighbors], axis=0, weights=weights) - X[i]

            # alignment = np.mean(V[neighbors], axis=0) - V[i]
            # cohesion = np.mean(X[neighbors], axis=0) - X[i]
            separation = np.sum((X[i] - X[neighbors]) /
                                (np.linalg.norm(X[i] - X[neighbors], axis=1)[:, None] ** 2 + 1e-9), axis=0)
            global_pull = gbest - X[i]

            new_V[i] = (w * V[i] +
                        alignment_weight * alignment +
                        cohesion_weight * cohesion +
                        separation_weight * separation +
                        global_best_attraction * global_pull)

        V = np.clip(new_V, -vel_max, vel_max)
        X = np.clip(X + V, bounds[0], bounds[1])

        # ----- GA phase -----
        new_pop = []
        elite_indices = np.argsort(fitness)[:elite_count]
        for idx in elite_indices:
            new_pop.append(X[idx].copy())

        while len(new_pop) < population:
            p1 = tournament_select(X, fitness, population)
            p2 = tournament_select(X, fitness, population)
            c1, c2 = crossover(p1, p2, crossover_rate, dimensions)
            c1 = mutate(c1, dimensions, mutation_rate, sigma, bounds)
            c2 = mutate(c2, dimensions, mutation_rate, sigma, bounds)
            new_pop.extend([c1, c2])

        X = np.array(new_pop[:population])

        # ----- Evaluate -----
        fitness = evaluate(X, function)
        best_index = np.argmin(fitness)
        if fitness[best_index] < gbest_val:
            gbest_val = fitness[best_index]
            gbest = X[best_index].copy()

        best_hist.append(gbest_val)

    return gbest, gbest_val, best_hist
