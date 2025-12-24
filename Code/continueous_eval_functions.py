import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sphere function
def sphere(x): 
    return np.sum(x**2)

# Rosenbrock function 
def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1) ** 2)

# Rastrigin function 
def rastrigin(x, A = 10):
    x = np.array(x)
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 + np.pi * x))


# Create a grid for 2D plotting
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Compute function values
Z_sphere = np.array([[sphere(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)] 
                     for row_x, row_y in zip(X, Y)])
Z_rastrigin = np.array([[rastrigin(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)] 
                        for row_x, row_y in zip(X, Y)])
Z_rosenbrock = np.array([[rosenbrock(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)] 
                         for row_x, row_y in zip(X, Y)])

# Plotting function
def plot_function(X, Y, Z, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    plt.show()

# Plot each function
plot_function(X, Y, Z_sphere, 'Sphere Function')
plot_function(X, Y, Z_rastrigin, 'Rastrigin Function')
plot_function(X, Y, Z_rosenbrock, 'Rosenbrock Function')