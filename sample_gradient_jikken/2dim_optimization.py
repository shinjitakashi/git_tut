import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


N = 20
x1 = np.linspace(-5,9,N)
x2 = np.linspace(-5,9,N)

x1, x2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(x1), np.ravel(x2)]

m = 2 #dimension

mean = np.zeros(m)
sigma = np.eye(m)

def y1(x1: np.array, x2: np.array) -> np.array:
    return -4*(x1**2-16)*(x1+4) + (x2+4)**2*(x2-4) #1 3

def y2(x1: np.array, x2: np.array) -> np.array:
    return (x1**3+x1**2)*(x1-4) + (-4*(x2**2-16)*(x2+4)) #2 1

def y3(x1: np.array, x2: np.array) -> np.array:
    return  (x1+4)**2*(x1-4) + (x2**3+x2**2)*(x2-4) #3 2

def objective_function(x1: np.array, x2: np.array) -> np.array:
    return y1(x1, x2) + y2(x1, x2) + y3(x1, x2)

y_plot = objective_function(x1, x2)
print(y_plot.shape)
y_plot = y_plot.reshape(x1.shape)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x1", fontsize=18)
ax.set_ylabel("x2", fontsize=18)
ax.set_zlabel("objective function", fontsize=18)



ax.plot_wireframe(x1, x2, y_plot)

plt.show()