from GA import ga
from PSO import pso
from FIPS import fips
from HybridPSO import hybrid_pso
from HybridGABoids import hybrid_ga_boids
from utils import plot_graph
import matplotlib.pyplot as plt
import numpy as np
from hyperparameters import DIMENSIONS


from hyperparameters import (
    ga_hyperparameters,
    pso_hyperparameters,
    hybrid_pso_hyperparameters,
    fips_hyperparameters, 
    ga_boids_hyperparameters
)
def run_continueous_eval_function(func_type: str, func, hyperparameters: dict, debug = False):
    eval_functions = ['sphere', 'rosenbrock', 'rastrigin']
    performances = []
    for f in eval_functions:
        _, best_val, best_hist = func(hyperparameters, f)
        if debug:
            print(f"{func_type} {f} Best Value: ", best_val)
        performances.append(best_val)
        plot_graph(best_hist, "Generation", "Value", f"{func_type} {f} Best value per Generation", True)
    # Sphere performance, rosenbrock performance, rastrigin performance
    return performances

def run_func_multiple_times(func_type: str, func, hyperparameters: dict, n: int, debug = False):
    perf_l = []
    perf_histories = []
    eval_func = 'rastrigin'
    for i in range(n):
        _, perf, perf_hist = func(hyperparameters, eval_func)
        if debug:
            print(f'{i} | {perf}')
        perf_l.append(perf)
        perf_histories.append(perf_hist)
    # Write values to a text file and tab it properly
    titles = ['Perf']
    with open(f'outputs/{func_type}_{eval_func}_{n}_D{DIMENSIONS}.txt', 'w') as f:
        # Write header
        #f.write(f"{titles[0]:<30}\n")
        # Write rows
        for a in perf_l:
            f.write(f"{a:<30}\n")
    # Plot convergence graphs
    for i in range(len(perf_histories)):
        plt.plot(perf_histories[i], color='gray', alpha=0.3)
    mean_values = np.mean(perf_histories, axis=0)

    # Plot the mean line (dark and thick)
    plt.plot(mean_values, color='black', linewidth=3, label='Mean')

    plt.title(f'{func_type} fitness over generations')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(f'plots/{func_type}_{eval_func}_{n}_D{DIMENSIONS}')
    plt.close()

if __name__ == "__main__":

    run_func_multiple_times("GA-Boids", hybrid_ga_boids, ga_boids_hyperparameters, 50, True)   
    #run_continueous_eval_function("GA", ga, ga_hyperparameters, True)
    # run_continueous_eval_function("PSO", pso, pso_hyperparameters, True)
    # run_continueous_eval_function("FIPS", fips, fips_hyperparameters, True)
    # run_continueous_eval_function("Hybrid PSO", hybrid_pso, hybrid_pso_hyperparameters, True)
    # run_continueous_eval_function("GA-Boids", hybrid_ga_boids, ga_boids_hyperparameters, True)


    # i love isa :)