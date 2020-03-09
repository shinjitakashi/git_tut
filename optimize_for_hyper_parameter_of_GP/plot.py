import numpy as np
import matplotlib.pyplot as plt

a = [[],[],[]]

for i in range(3):
    a[i] = np.loadtxt('./result_data/24.593/grad_array_0'+ str(i)+'.dat')
for i in range(3):
    print(a[i][-5:-1])  
for i in range(3):
    plt.plot(np.arange(0, len(a[i])), a[i])
    # plt.ylim(-3,0)
    plt.show()