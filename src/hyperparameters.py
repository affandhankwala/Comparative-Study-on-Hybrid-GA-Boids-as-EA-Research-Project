POPULATION = 100
BOUNDS = (-5.12, 5.12)
GENERATIONS = 300
DIMENSIONS = 100

ga_hyperparameters = {
    'POPULATION' : 150,
    'GENERATIONS' : GENERATIONS,
    'DIMENSIONS' : DIMENSIONS,
    'BOUNDS' : BOUNDS,
    'CROSSOVER_RATE' : 0.9,
    'MUTATION_RATE' : 0.11,      
    'ELITE_COUNT' : 2,
    'SIGMA' : 0.1 * (BOUNDS[1] - BOUNDS[0])
}

pso_hyperparameters = {
    'POPULATION' : POPULATION,
    'GENERATIONS' : GENERATIONS,
    'DIMENSIONS' : DIMENSIONS,
    'BOUNDS' : BOUNDS,
    'W' : 0.3,          # Inertia weight
    'C1' : 2,         # Cognitive coefficient
    'C2' : 2,         # Social coefficient
    'VEL_MAX' : 0.1 * (BOUNDS[1] - BOUNDS[0])  # Velocity clamping
}

fips_hyperparameters = {
    # Shared hyperparameters
    'POPULATION' : POPULATION,
    'GENERATIONS' : GENERATIONS,
    'DIMENSIONS' : DIMENSIONS,
    'BOUNDS' : BOUNDS,
    'VEL_MAX' : 0.1 * (BOUNDS[1] - BOUNDS[0]),

    'W' : 0.6,     # inertia weight
    'C' : 1.5,     # total acceleration weight (distributed among neighbors)
    'K' : 2       # number of neighbors per particle
}

hybrid_pso_hyperparameters = {
    # Shared hyperparameters
    'POPULATION' : POPULATION,
    'GENERATIONS' : GENERATIONS,
    'DIMENSIONS' : DIMENSIONS,
    'BOUNDS' : BOUNDS,
    'VEL_MAX' : 0.1 * (BOUNDS[1] - BOUNDS[0]),

    'W' : 0.3,
    'C1' : 1.5,
    'C2' : 2,
    'MUTATION_RATE' : 0.2,
    'SIGMA' : 0.05 * (BOUNDS[1] - BOUNDS[0]),
    'STAGNATION_LIMIT' : 30  # generations without improvement before triggering mutation
}

ga_boids_hyperparameters = {
    'POPULATION' : POPULATION,
    'GENERATIONS' : GENERATIONS,
    'DIMENSIONS' : DIMENSIONS,
    'BOUNDS' : BOUNDS,

    # Boids + GA parameters
    'W' : 0.3,           # inertia for velocity
    'ALIGNMENT_WEIGHT' : 0.3,
    'COHESION_WEIGHT' : 0.3,
    'SEPARATION_WEIGHT' : 0.3,
    'GLOBAL_BEST_ATTRACTION' : 0.8,
    'VEL_MAX' : 0.1 * (BOUNDS[1] - BOUNDS[0]),
    'NEIGHBOR_RADIUS' : 0.2,
    'CROSSOVER_RATE' : 0.9,
    'MUTATION_RATE' : 0.11,
    'SIGMA' : 0.05 * (BOUNDS[1] - BOUNDS[0]),
    'ELITE_COUNT' : 2
}