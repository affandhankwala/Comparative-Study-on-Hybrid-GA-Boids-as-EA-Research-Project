import matplotlib.pyplot as plt
import numpy as np
from GA import evaluate

def plot_graph (data: list, xlabel: str, ylabel: str, title: str, save = False):
    x = list(range(1, len(data) + 1))
    plt.figure(figsize= (8,5))
    plt.plot(x, data, linewidth = 2, color = "blue")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save: 
        plt.savefig(f'plots/{title}')
    else:
        plt.show()
    plt.close()

def save_res(func, best_sphere, best_rosenbrock, best_rastrigin,
             best_sphere_hyp, best_rosenbrock_hyp, best_rastrigin_hyp, 
             time):
    with open(f'tuned_performances/{func}.txt', 'w') as f: 
        f.write(f'Best sphere: {best_sphere} \n')
        f.write(str(best_sphere_hyp))
        f.write('\n')

        f.write(f'Best Rosenbrock: {best_rosenbrock} \n')
        f.write(str(best_rosenbrock_hyp))
        f.write('\n')

        f.write(f'Best Rastrigin: {best_rastrigin} \n')
        f.write(str(best_rastrigin_hyp))
        f.write('\n')

        f.write(f"Tuning time: {time} \n")
