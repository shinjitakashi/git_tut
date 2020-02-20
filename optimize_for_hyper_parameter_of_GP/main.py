import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LN
import networkx as nx
import copy

from Agent import *


if __name__ == '__main__':
    #Parameters
    #Number of agents
    N = 5

    #Number of dimensions of the decision variable
    n = 5

    #Coefficient of decision of stepsize : a(t) = a / t
    stepsize = 0.08
            
    # Coefficient of the edge weight  w_if = wc / max_degree
    wc = 0.6

    #Number of iterations
    iteration = 30

    #Coefficient of decision of stepsize : E_ij(t) = E(t) = eventtrigger / (t+1)
    eventtrigger = [0, 1, 5]

    # Randomization seed
    np.random.seed(9)

    #======================================================================#
    #Communication Graph
    A = np.array(
    [[1, 1, 0, 1, 1],
     [1, 1, 0, 1, 0],
     [0, 0, 1, 1, 1],
     [1, 1, 1, 1, 0],
     [1, 0, 1, 0, 1]
     ])

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

    find_n = 150 # 既知の点の数
    x0 = np.random.uniform(0,200,find_n) # 既知の点
    y0 = y(x0) + np.random.normal(0,0.5,find_n)
    param0 = [[11.5,15.2,0.3],[14.3,1.8,0.6],[1.2,13.2,1.7],[10.8,8.3,0.2],[7.4,9.3,3]] # パラメータの初期値
    #param0 =[[1.5,0.4,2.7],[2.3,1.8,1.3],[3.2,1.2,0.7]]
    # [[1.5,0.4,2.7],[2.3,2.8,1.3],[3.2,1.2,1.7]]
    # [[1.5,3.4,2.7],[2.3,2.8,1.3],[3.2,1.2,1.7]] これいい！！
    """
    0に関して，制約をつけないとリプシッツ連続性を言えなくなるため，
    boundにて，制約をつけペナルティ関数法を用いてアルゴリズムの実装を行う．
    """
    bound = [[1e-2,1e2],[1e-2,1e2],[1e-2,1e2]] # 下限上限
    #kernel = Kernel(param0[0],bound)
    
    x1 = np.linspace(0,200,400) #予測用のデータ
    
    # alone_gp = Gausskatei(kernel) 
    # alone_gp.gakushuu(x0,y0)
    
    # for i in [0,1]:
    #     if(i):
    #         alone_gp.saitekika(xgit p0,y0,1000) # パラメータを調整する
    #     plt.subplot(211+i)
    #     plt.plot(x0,y0,'. ')
    #     mu,std = alone_gp.yosoku(x1)
    #     plt.plot(x1,y(x1),'--r')
    #     plt.plot(x1,mu,'g')
    #     plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
    #     plt.title('a=%.3f, s=%.3f, w=%.3f'%tuple(alone_gp.kernel.param))
    # plt.tight_layout()
    # plt.show()

    xd = []
    yd = []

    for i in range(N):
        tmp_xd = np.random.uniform(0,200,30)
        xd.append(tmp_xd)
        yd.append(y(tmp_xd)+np.random.normal(0,0.5,30))
    
    gp = []
    multi_kernel = []
    for i in range(N):
        multi_kernel.append(Kernel(param0[i], bound))
    for i in range(N):
        gp.append(GausskateiWithMyTheory(multi_kernel[i], N, i, P, xd[i], yd[i]))
        gp[i].gakushuu(x0, y0)

    normalize_error = []

    tmp_normalize_error = [0] * N
    
    for q in range(N):
        mu,std = gp[q].yosoku(x1)
        seikai = y(x1)
        for j in range(len(x1)):
            tmp_normalize_error[q] += np.abs(mu[j] - seikai[j])**2   
                
    initial_error = tmp_normalize_error[0] + tmp_normalize_error[1] + tmp_normalize_error[2]


    for i in [0,1]:
        multi_gp = copy.deepcopy(gp)
        if (i):
            #初期時刻での情報交換
            for i in range(N):
                for j in range(N):
                    if i!=j and A[i][j]==1:
                        multi_gp[j].receive(multi_gp[i].send(j),i)
                multi_gp[i].Hp_send[i] = multi_gp[i].theta
            
            for t in range(iteration):
                if t==0:
                    normalize_error.append(initial_error)
                
                if (t%100)==0:            
                    print(str(t) + '回目')
                for i in range(N):
                    for j in range(N):
                        if i!=j and A[i][j]==1:
                            multi_gp[j].receive(multi_gp[i].send(j), i)

                for i in range(N):
                    multi_gp[i].saitekika(t+1)

                tmp_normalize_error = [0]*N

                for q in range(N):
                    multi_gp[q].gakushuu(x0,y0)
                    mu,std = multi_gp[q].yosoku(x1)
                    seikai = y(x1)
                    for j in range(len(x1)):
                        tmp_normalize_error[q] += np.abs(mu[j] - seikai[j])**2   
                
                normalize_error.append(tmp_normalize_error[0] + tmp_normalize_error[1] + tmp_normalize_error[2])
        
        for d in range(N):
            multi_gp[d].gakushuu(x0,y0)
            plt.subplot(321+d)
            plt.plot(x0,y0,'. ')
            mu,std = multi_gp[d].yosoku(x1)
            plt.plot(x1,y(x1),'--r')
            plt.plot(x1,mu,'g')
            plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
            plt.title('a=%.3f, s=%.3f, w=%.3f'%tuple(multi_gp[d].kernel.param) + str(multi_gp[d].name))
            plt.tight_layout()

        plt.show()    

        if (i):
            plt.title('normalize_error')
            plt.plot(np.arange(0,iteration+1), np.array(normalize_error)/initial_error)
            plt.yscale('log')
            plt.show()

