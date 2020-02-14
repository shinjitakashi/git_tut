import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LN
import networkx as nx
import copy

from Agent import *


if __name__ == '__main__':
    #Parameters
    #Number of agents
    N = 3

    #Number of dimensions of the decision variable
    n = 3

    #Coefficient of decision of stepsize : a(t) = a / t
    stepsize = 0.008
            
    # Coefficient of the edge weight  w_if = wc / max_degree
    wc = 0.8

    #Number of iterations
    iteration = 1000

    #Coefficient of decision of stepsize : E_ij(t) = E(t) = eventtrigger / (t+1)
    eventtrigger = [0, 1, 5]

    # Randomization seed
    np.random.seed(9)

    #======================================================================#
    #Communication Graph
    A = np.array(
    [[1, 1, 1],
     [1, 1, 0],
     [1, 0, 1]])

    G = nx.from_numpy_matrix(A)

    # Weighted Stochastic Matrix P
    a = np.zeros(N)

    for i in range(N):
        a[i] = copy.copy(wc / nx.degree(G)[i])

    P = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i != j and A[i][j] == 1:
                a_ij = min(a[i], a[j])
                P[i][j] = copy.copy(a_ij)
                P[j][i] = copy.copy(a_ij)

    for i in range(N):
        sum = 0.0
        for j in range(N):
            sum += P[i][j]
        P[i][i] = 1.0 - sum
    #======================================================================#
    def y(x): # 実際の関数
        return 5*np.sin(np.pi/15*x)*np.exp(-x/50)

    n = 100 # 既知の点の数
    x0 = np.random.uniform(0,100,n) # 既知の点
    y0 = y(x0) + np.random.normal(0,1,n)
    param0 = [2,0.4,1.5] # パラメータの初期値
    bound = [[1e-2,1e2],[1e-2,1e2],[1e-2,1e2]] # 下限上限
    kernel = Kernel(param0,bound)
    
    x1 = np.linspace(0,100,200) #予測用のデータ
    
    alone_gp = Gausskatei(kernel) 
    alone_gp.gakushuu(x0,y0)

    gp = []
    for i in range(N):
        gp.append(GausskateiWithMyTheory(kernel, N, i, P))
        gp[i].gakushuu(x0, y0)
    plt.figure(figsize=[5,8])
    
    for i in [0,1]:
        if(i):
            alone_gp.saitekika(x0,y0,1000) # パラメータを調整する
        plt.subplot(211+i)
        plt.plot(x0,y0,'. ')
        mu,std = alone_gp.yosoku(x1)
        plt.plot(x1,y(x1),'--r')
        plt.plot(x1,mu,'g')
        plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
        plt.title('a=%.3f, s=%.3f, w=%.3f'%tuple(alone_gp.kernel.param))
    plt.tight_layout()
    plt.show()
