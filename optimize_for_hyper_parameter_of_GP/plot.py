import numpy as np
import matplotlib.pyplot as plt


a = np.loadtxt('./result_data/147.662/grad_array_01.dat')

print(len(a))
plt.plot(np.arange(0, len(a)), a)
plt.ylim(-3,0)
plt.show()