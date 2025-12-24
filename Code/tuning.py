from GA import ga
from PSO import pso
from FIPS import fips
from HybridPSO import hybrid_pso
from HybridGABoids import hybrid_ga_boids
from main import run_continueous_eval_function
from hyperparameters import GENERATIONS, DIMENSIONS, BOUNDS
from utils import save_res

import numpy as np
import time

# Hyperparameter ranges
#POPS = [50, 100, 150]
POPS = [150]
#W = [0.9, 0.6, 0.3]
W = [0.3]
A_C_S_W = [0.3, 0.5, 0.8]
G_B_A = [0.8, 1.4, 2]
#C_R = [0.8, 0.9, 1]
C_R = [0.9]
M_R = [0.02, 0.11, 0.2]
N_R = [0.2, 0.35, 0.5]
E_C = [2, 3, 5]
C = [1, 1.5, 2]
A_C = [1.5, 2.05, 3]
K = [2, 5, 10]
S_L = [30, 50, 100]

def tune_ga():
    # Total ranges
    total_tests = (
        len(POPS) * len(C_R) * len(M_R) * len(E_C)
    )
    best_sphere, best_rosenbrock, best_rastrigin = 100000, 10000, 10000
    best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp = None, None, None
    test_no = 1
    start_time = time.time()

    generations = GENERATIONS
    dimensions = DIMENSIONS
    bounds = BOUNDS
    vel_max = 0.1 * (BOUNDS[1] - BOUNDS[0])
    sigma = 0.05 * (BOUNDS[1] - BOUNDS[0])
     
    for p in range(len(POPS)):
        for c_r in range(len(C_R)):
            for m_r in range(len(M_R)):
                for e_c in range(len(E_C)):
                    # Define hyperparameters
                    hyperparameters = {
                        'POPULATION' : POPS[p],
                        'GENERATIONS' : generations,
                        'DIMENSIONS' : dimensions,
                        'BOUNDS' : bounds, 
                        'VEL_MAX' : vel_max,
                        'CROSSOVER_RATE' : C_R[c_r],
                        'MUTATION_RATE' : M_R[m_r],
                        'SIGMA' : sigma,
                        'ELITE_COUNT' : E_C[e_c]
                    }
                    # Tracking
                    iteration = f'GA: {p}.{c_r}.{m_r}.{e_c}'                                            
                    print(f'{iteration}  |  {test_no}/{total_tests}  |  {round(test_no/total_tests * 100, 2)}%')
                    test_no += 1

                    perf = run_continueous_eval_function("GA", ga, hyperparameters)
                    sp_p, rosen_p, rast_p = perf[0], perf[1], perf[2]
                    if sp_p < best_sphere:
                        best_sphere = sp_p
                        best_sphere_hyp = hyperparameters
                    if rosen_p < best_rosenbrock:
                        best_rosenbrock = rosen_p
                        best_rosenbrock_hyp = hyperparameters
                    if rast_p < best_rastrigin:
                        best_rastrigin = rast_p
                        best_rastrigin_hyp = hyperparameters
    end_time = time.time()
    # Save bests to text file
    save_res("GA", best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp,
             end_time - start_time)

def tune_pso():
    # Total ranges
    total_tests = (
        len(POPS) * len(W) * len(C) * len(C)
        )
    best_sphere, best_rosenbrock, best_rastrigin = 100000, 10000, 10000
    best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp = None, None, None
    test_no = 1
    start_time = time.time()

    generations = GENERATIONS
    dimensions = DIMENSIONS
    bounds = BOUNDS
    vel_max = 0.1 * (BOUNDS[1] - BOUNDS[0])
     
    for p in range(len(POPS)):
        for w in range(len(W)):
            for c1 in range(len(C)):
                for c2 in range(len(C)):
                    # Define hyperparameters
                    hyperparameters = {
                        'POPULATION' : POPS[p],
                        'GENERATIONS' : generations,
                        'DIMENSIONS' : dimensions,
                        'BOUNDS' : bounds, 
                        'VEL_MAX' : vel_max,
                        'W': W[w],
                        'C1' : C[c1],
                        'C2' : C[c2]
                    }
                    # Tracking
                    iteration = f'PSO: {p}.{w}.{c1}.{c2}'                                            
                    print(f'{iteration}  |  {test_no}/{total_tests}  |  {round(test_no/total_tests * 100, 2)}%')
                    test_no += 1

                    perf = run_continueous_eval_function("PSO", pso, hyperparameters)
                    sp_p, rosen_p, rast_p = perf[0], perf[1], perf[2]
                    if sp_p < best_sphere:
                        best_sphere = sp_p
                        best_sphere_hyp = hyperparameters
                    if rosen_p < best_rosenbrock:
                        best_rosenbrock = rosen_p
                        best_rosenbrock_hyp = hyperparameters
                    if rast_p < best_rastrigin:
                        best_rastrigin = rast_p
                        best_rastrigin_hyp = hyperparameters
    end_time = time.time()
    # Save bests to text file
    save_res("PSO", best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp,
             end_time - start_time)

def tune_fips():
    # Total ranges
    total_tests = (
        len(POPS) * len(W) * len(A_C) * len(K)
    )
    best_sphere, best_rosenbrock, best_rastrigin = 100000, 10000, 10000
    best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp = None, None, None
    test_no = 1
    start_time = time.time()

    generations = GENERATIONS
    dimensions = DIMENSIONS
    bounds = BOUNDS
    vel_max = 0.1 * (BOUNDS[1] - BOUNDS[0])
     
    for p in range(len(POPS)):
        for w in range(len(W)):
            for a_c in range(len(A_C)):
                for k in range(len(K)):
                    # Define hyperparameters
                    hyperparameters = {
                        'POPULATION' : POPS[p],
                        'GENERATIONS' : generations,
                        'DIMENSIONS' : dimensions,
                        'BOUNDS' : bounds, 
                        'VEL_MAX' : vel_max,
                        'W' : W[w],
                        'C' : A_C[a_c],
                        'K' : K[k]
                    }
                    # Tracking
                    iteration = f'FIPS: {p}.{w}.{a_c}.{k}'                                            
                    print(f'{iteration}  |  {test_no}/{total_tests}  |  {round(test_no/total_tests * 100, 2)}%')
                    test_no += 1

                    perf = run_continueous_eval_function("FIPS", fips, hyperparameters)
                    sp_p, rosen_p, rast_p = perf[0], perf[1], perf[2]
                    if sp_p < best_sphere:
                        best_sphere = sp_p
                        best_sphere_hyp = hyperparameters
                    if rosen_p < best_rosenbrock:
                        best_rosenbrock = rosen_p
                        best_rosenbrock_hyp = hyperparameters
                    if rast_p < best_rastrigin:
                        best_rastrigin = rast_p
                        best_rastrigin_hyp = hyperparameters
    end_time = time.time()
    # Save bests to text file
    save_res("FIPS", best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp,
             end_time - start_time)

def tune_hybrid_pso():
    # Total ranges
    total_tests = (
        len(POPS) * len(W) * len(C) * len(C) * len(M_R) * len(S_L)
    )
    best_sphere, best_rosenbrock, best_rastrigin = 100000, 10000, 10000
    best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp = None, None, None
    test_no = 1
    start_time = time.time()

    generations = GENERATIONS
    dimensions = DIMENSIONS
    bounds = BOUNDS
    vel_max = 0.1 * (BOUNDS[1] - BOUNDS[0])
    sigma = 0.05 * (BOUNDS[1] - BOUNDS[0])

    for p in range(len(POPS)):
        for w in range(len(W)):
            for m_r in range(len(M_R)):
                for c1 in range(len(C)):
                    for c2 in range(len(C)):
                        for s_l in range(len(S_L)):
                            # Define hyperparameters
                            hyperparameters = {
                                'POPULATION' : POPS[p],
                                'GENERATIONS' : generations,
                                'DIMENSIONS' : dimensions,
                                'BOUNDS' : bounds, 
                                'VEL_MAX' : vel_max,
                                'W' : W[w],
                                'C1' : C[c1],
                                'C2' : C[c2],
                                'MUTATION_RATE' : M_R[m_r],
                                'STAGNATION_LIMIT' : S_L[s_l],
                                'SIGMA': sigma
                            }
                    # Tracking
                    iteration = f'Hybrid PSO: {p}.{w}.{m_r}.{c1}.{c2}.{s_l}'                                            
                    print(f'{iteration}  |  {test_no}/{total_tests}  |  {round(test_no/total_tests * 100, 2)}%')
                    test_no += 1

                    perf = run_continueous_eval_function("Hybrid PSO", hybrid_pso, hyperparameters)
                    sp_p, rosen_p, rast_p = perf[0], perf[1], perf[2]
                    if sp_p < best_sphere:
                        best_sphere = sp_p
                        best_sphere_hyp = hyperparameters
                    if rosen_p < best_rosenbrock:
                        best_rosenbrock = rosen_p
                        best_rosenbrock_hyp = hyperparameters
                    if rast_p < best_rastrigin:
                        best_rastrigin = rast_p
                        best_rastrigin_hyp = hyperparameters
    end_time = time.time()
    # Save bests to text file
    save_res("Hybrid_PSO", best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp,
             end_time - start_time)
      
def tune_ga_boids():
    # Total ranges
    total_tests = (
        len(POPS) * len(W) * len(A_C_S_W) * len(A_C_S_W) * len(A_C_S_W) * len(G_B_A) *
        len(C_R) * len(M_R) * len(N_R) * len(E_C)
    )
    best_sphere, best_rosenbrock, best_rastrigin = 100000, 1000000, 100000
    best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp = None, None, None
    test_no = 1
    start_time = time.time()

    
    generations = GENERATIONS
    dimensions = DIMENSIONS
    bounds = BOUNDS
    vel_max = 0.1 * (BOUNDS[1] - BOUNDS[0])
    sigma = 0.05 * (BOUNDS[1] - BOUNDS[0])

    # Tune rest of parameters
    for p in range(len(POPS)):
        for w in range(len(W)):
            for a_w in range(len(A_C_S_W)):
                for c_w in range(len(A_C_S_W)):
                    for s_w in range(len(A_C_S_W)):
                        for g_b_a in range(len(G_B_A)):
                            for c_r in range(len(C_R)):
                                for m_r in range(len(M_R)):
                                    for n_r in range(len(N_R)):
                                        for e_c in range(len(E_C)):
                                            # Define hyperparameters
                                            hyperparameters = {
                                                'POPULATION' : POPS[p],
                                                'GENERATIONS' : generations,
                                                'DIMENSIONS' : dimensions,
                                                'BOUNDS' : bounds, 
                                                'W' : W[w],
                                                'ALIGNMENT_WEIGHT' : A_C_S_W[a_w],
                                                'COHESION_WEIGHT' : A_C_S_W[c_w],
                                                'SEPARATION_WEIGHT' : A_C_S_W[s_w],
                                                'GLOBAL_BEST_ATTRACTION' : G_B_A[g_b_a],
                                                'VEL_MAX' : vel_max,
                                                'NEIGHBOR_RADIUS' : N_R[n_r],
                                                'CROSSOVER_RATE' : C_R[c_r],
                                                'MUTATION_RATE' : M_R[m_r],
                                                'SIGMA' : sigma,
                                                'ELITE_COUNT' : E_C[e_c]
                                            }
                                            # Tracking
                                            iteration = f'{p}.{w}.{a_w}.{c_w}.{s_w}.{g_b_a}.{c_r}.{m_r}.{n_r}.{e_c}'                                            
                                            print(f'{iteration}  |  {test_no}/{total_tests}  |  {round(test_no/total_tests * 100, 2)}%')
                                            test_no += 1

                                            perf = run_continueous_eval_function("GA-Boids", hybrid_ga_boids, hyperparameters)
                                            sp_p, rosen_p, rast_p = perf[0], perf[1], perf[2]
                                            if sp_p < best_sphere:
                                                best_sphere = sp_p
                                                best_sphere_hyp = hyperparameters
                                            if rosen_p < best_rosenbrock:
                                                best_rosenbrock = rosen_p
                                                best_rosenbrock_hyp = hyperparameters
                                            if rast_p < best_rastrigin:
                                                best_rastrigin = rast_p
                                                best_rastrigin_hyp = hyperparameters
    end_time = time.time()
    # Save bests to text file
    save_res("GA_Boids", best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp,
             end_time - start_time)
    


def test():
    ga_boids_hyperparameters = {
        'POPULATION' : POPULATION,
        'GENERATIONS' : GENERATIONS,
        'DIMENSIONS' : DIMENSIONS,
        'BOUNDS' : BOUNDS,

        # Boids + GA parameters
        'W' : 0.7,           # inertia for velocity
        'ALIGNMENT_WEIGHT' : 0.4,
        'COHESION_WEIGHT' : 0.4,
        'SEPARATION_WEIGHT' : 0.6,
        'GLOBAL_BEST_ATTRACTION' : 1.0,
        'VEL_MAX' : 0.1 * (BOUNDS[1] - BOUNDS[0]),
        'NEIGHBOR_RADIUS' : 0.3,
        'CROSSOVER_RATE' : 0.9,
        'MUTATION_RATE' : 0.05,
        'SIGMA' : 0.05 * (BOUNDS[1] - BOUNDS[0]),
        'ELITE_COUNT' : 2
    }
    with open('tuned_performances/test.txt', 'w') as f:
        f.write('begin test \n')
        f.write(str(ga_boids_hyperparameters))

# Run tuning
# tune_ga()
# tune_pso()
# tune_fips()
# tune_hybrid_pso()
tune_ga_boids()
                                            
