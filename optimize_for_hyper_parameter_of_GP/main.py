import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LN
import networkx as nx
import copy
import os

from Agent import *


path = os.path.dirname(os.path.abspath(__file__))

def main(normalize_error, iteration, param):
    global save_dir

    makeDir(normalize_error)
    save_dir = path+'/result_data/'+str(round(normalize_error, 3))+'/'
    np.savetxt(save_dir+'iteration.dat',[iteration])
    np.savetxt(save_dir+'initial_param.dat', param)

def makeDir(normalize_error):
    if not os.path.isdir(path+'/result_data/'+str(round(normalize_error, 3))):
        os.mkdir(path+'/result_data/'+str(round(normalize_error, 3)))



if __name__ == '__main__':
    #Parameters
    #Number of agents
    N = 5

    #Number of dimensions of the decision variable
    n = 5

    #Coefficient of decision of stepsize : a(t) = a / t
    
    #良い
    # stepsize = [0.01, 0.01, 0.01]


    stepsize = [0.01, 0.0001, 0.0001]
    #ペナルティ関数法を用いる時のステップサイズ

    constant_for_time = [10000, 10000, 10000]
    # constant_for_time = [100, 100, 100]
    
    # Coefficient of the edge weight  w_if = wc / max_degree
    wc = 0.8

    #Number of iterations
    iteration = 1000

    #Coefficient of decision of stepsize : E_ij(t) = E(t) = eventtrigger / (t+1)
    eventtrigger = [0, 1, 5]

    event_size = 3

    # Randomization seed
    np.random.seed(1)

    #======================================================================#
    #Communication Graph
    A = np.array(
    [[1, 0, 0, 1, 1],
     [0, 1, 1, 1, 1],
     [0, 1, 1, 0, 1],
     [1, 1, 0, 1, 0],
     [1, 1, 1, 0, 1]
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

    find_n = 100 # 既知の点の数
    x0 = np.random.uniform(0,100,find_n) # 既知の点
    y0 = y(x0) + np.random.normal(0,0.1,find_n)
    # param0 = [[4,2.3,0.05],[0.4,3.4,0.5],[1.6,1.5,0.56],[2.8,2.3,0.8],[1.5,3.3,1.3]] # パラメータの初期値
    
    #ペナルティ関数法
    # param0 = [[1.05,0.3,1.3],[2.3,2.4,3.0],[0.2,0.42,1.06],[1.8,2.3,0.2],[0.4,0.3,2.3]]
    
    param0 = [[1.5,0.5,0.13],[1.1,0.4,0.5],[1.2,0.42,0.6],[1.8,2.3,0.2],[0.7,7.3,1.3]] # パラメータの初期値

    # param0 = [[1.5,0.5,1.3],[1.1,0.4,4.0],[1.2,0.42,2.6],[1.8,2.3,1.2],[0.7,7.3,2.3]] # パラメータの初期値
    """
    0に関して，制約をつけないとリプシッツ連続性を言えなくなるため，
    boundにて，制約をつけペナルティ関数法を用いてアルゴリズムの実装を行う．
    """
    bound = [[1e-2,5e1],[1e-2,5e1],[1e-2,5e1]] # 下限上限
    #kernel = Kernel(param0[0],bound)
    
    x1 = np.linspace(0,100,200) #予測用のデータ
    
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

    #最適化用のデータ
    for i in range(N):
        tmp_xd = np.random.uniform(0,100,20)
        xd.append(tmp_xd)
        yd.append(y(tmp_xd)+np.random.normal(0,0.1,20))
    
    gp = []
    multi_kernel = []
    for i in range(N):
        multi_kernel.append(Kernel(param0[i], bound))
    for i in range(N):
        gp.append(GausskateiWithMyTheory(multi_kernel[i], N, i, P, xd[i], yd[i], eventtrigger, stepsize, constant_for_time))
        gp[i].gakushuu(x0, y0)

    normalize_error = [[],[],[]]
    objective_function_array = [[],[],[]]

    grad_array = [[[],[],[]],[[],[],[]],[[],[],[]]]

    tmp_normalize_error = [0] * N

    for q in range(N):
        mu,std = gp[q].yosoku(x1)
        seikai = y(x1)
        for j in range(len(x1)):
            tmp_normalize_error[q] += np.abs(mu[j] - seikai[j])**2   

    initial_error = 0
    for q in range(N):
        initial_error += tmp_normalize_error[q]

    #勾配の変化を示すarray
    #gradient_array = [0]*range(eventtrigger)

    terminal_count = [[],[],[]]

    for i in [0,1]:
        multi_gp = copy.deepcopy(gp)

        if (i):
            for e in range(len(eventtrigger)):
                count = 0

                print('E=', str(multi_gp[0].param_for_event[e]))
                #初期時刻での情報交換
                for i in range(N):
                    for j in range(N):
                        if i!=j and A[i][j]==1:
                            multi_gp[j].receive(multi_gp[i].send(j),i)
                    multi_gp[i].Hp_send[i] = multi_gp[i].theta

                for t in range(iteration):

                    if t==0:
                        normalize_error[e].append(initial_error)

                    
                    for i in range(N):
                        for j in range(N):
                            if i!=j and A[i][j]==1:
                                if LN.norm(multi_gp[i].kernel.param-multi_gp[i].Hp_send[j], ord=1) > multi_gp[i].event_trigger(t+1, multi_gp[i].param_for_event[e]):
                                    count += 1

                                    multi_gp[j].receive(multi_gp[i].send(j), i)

                    tmp_grad = np.array([0,0,0], dtype='float64')
                    tmp_objective_function = 0

                    for i in range(N):
                        tmp_objective_function += multi_gp[i].ref_objective_function()
                        multi_gp[i].saitekika(t, e)
                        for j in range(len(multi_gp[i].kernel.param)):
                            tmp_grad[j] += multi_gp[i].grad[j]

                    
                    objective_function_array[e].append(tmp_objective_function)

                    for d in range(len(multi_gp[0].kernel.param)):
                        grad_array[e][d].append(tmp_grad[d])

                    tmp_normalize_error = [0]*N
                    sum_normalize_error = 0

                    
                    for q in range(N):
                        multi_gp[q].gakushuu(x0,y0)
                        mu,std = multi_gp[q].yosoku(x1)
                        seikai = y(x1)
                        for j in range(len(x1)):
                            tmp_normalize_error[q] += np.abs(mu[j] - seikai[j])**2   
                        sum_normalize_error += tmp_normalize_error[q]
                        
                    
                    normalize_error[e].append(sum_normalize_error)

                    if (t%100)==0:            
                        print(str(t) + '回目')
                        print('==========================')
                        print('objective function')
                        print(objective_function_array[e][-1])
                        print('==========================')
                        print('normalize error')
                        print(normalize_error[e][-1])
                        print('==========================')
                        for d in range(3):
                            print(grad_array[e][d][-1])
                
                terminal_count[e] = count
                
                if e==0:
                    main(normalize_error[e][-1], iteration, param0)

                plt.figure(figsize=(8,15))
                for d in range(N):
                    multi_gp[d].gakushuu(x0,y0)
                    plt.subplot(321+d)
                    plt.plot(x0,y0,'. ', label='observed data')
                    mu,std = multi_gp[d].yosoku(x1)
                    plt.plot(x1,y(x1),'--r', label='True')
                    plt.plot(x1,mu,'g', label='estimation')
                    plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
                    plt.title('a=%.3f, s=%.3f, w=%.3f, agent:'%tuple(multi_gp[d].kernel.param) + str(multi_gp[d].name))
                    plt.tight_layout()
                plt.savefig(os.path.join(save_dir,'Yosoku_E='+str(eventtrigger[e])+'.pdf'))
                
                print('==========================')
                print('objective function')
                print(objective_function_array[e][-1])
                print('==========================')
                print('normalize error')
                print(normalize_error[e][-1])
                print('==========================')
                for d in range(3):
                    print(grad_array[e][d][-1])


        if (i):

            color = ['g', 'b', 'r']
            label = ["E="+str(eventtrigger[0]), "E="+str(eventtrigger[1]), "E="+str(eventtrigger[2])]
            
            plt.figure(figsize=(7,7))
            plt.title('normalize_error')

            for e in range(len(eventtrigger)):
                plt.plot(np.arange(0,iteration+1), np.array(normalize_error[e])/initial_error, color=color[e], label=label[e])
            plt.yscale('log')
            plt.legend(loc='best')

            plt.xlabel('time t')
            plt.ylabel('normalize_error')

            plt.savefig(os.path.join(save_dir,'normalize_error'+'.pdf'))
            
            theta_array = ['theta1', 'theta2', 'theta3']

            
            """
                しきい値パラメータに対する各ハイパーパラメータの勾配をみたい
                1: jに対して，各ハイパーパラメータの勾配を見るために
            """
            for e in range(len(eventtrigger)):
                plt.figure(figsize=(7,7))
                for j in range(len(multi_gp[0].kernel.param)):
                    plt.plot(np.arange(0,iteration), grad_array[e][j], color=color[j], label=theta_array[j])
                plt.legend(loc='best')
                # plt.yscale(-100, 100)
                
                plt.xlabel('time t')
                plt.ylabel('grad')

                plt.ylim(-100,100)

                plt.savefig(os.path.join(save_dir, 'grad_for_'+label[e]+'.pdf'))

            plt.figure(figsize=(5,7))
            left = [0,1,2]

            plt.bar(left, terminal_count, tick_label=label)

            plt.ylabel('communication count')
            plt.xlabel('param for event-triggered communication')            

            for x, y in  zip(left, terminal_count):
                plt.text(x, y, y, ha='center', va='bottom')
            
            plt.savefig(os.path.join(save_dir,'communication_count'+'.pdf'))

        # if (i):
        #     plt.figure(figsize=(8,15))
        #     for d in range(N):
        #         x1 = np.linspace(0,200,400) #予測用のデータ

        #         multi_gp[d].kernel.param = param0[d]
        #         multi_gp[d].gakushuu(x0,y0)
        #         plt.subplot(321+d)
        #         plt.plot(x0,y0,'. ', label='observed data')
        #         mu,std = multi_gp[d].yosoku(x1)
                
        #         plt.plot(x1,y(x1),'--', color='r', label='True')
        #         plt.plot(x1,mu,'g', label='estimation')

        #         plt.legend(loc='best')

        #         plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
        #         plt.title('a=%.3f, s=%.3f, w=%.3f, agent:'%tuple(multi_gp[d].kernel.param) + str(multi_gp[d].name))
        #         plt.tight_layout()
        #     plt.savefig(os.path.join(save_dir,'initial.pdf'))
        #     # plt.show()