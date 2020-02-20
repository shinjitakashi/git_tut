import matplotlib.pyplot as plt
import numpy.linalg as LN
import networkx as nx
import copy

from Agent import *

if __name__=='__main__':
    def y(x): # 実際の関数
        return 5*np.sin(np.pi/15*x)*np.exp(-x/50)

    n = 100 # 既知の点の数
    x0 = np.random.uniform(0,100,n) # 既知の点
    y0 = y(x0) + np.random.normal(0,1,n)
    param0 = [3,0.6,0.5] # パラメータの初期値
    bound = [[1e-2,1e2],[1e-2,1e2],[1e-2,1e2]] # 下限上限
    kernel = Kernel(param0,bound)
    x1 = np.linspace(0,100,200)
    gp = Gausskatei(kernel)
    gp.gakushuu(x0,y0) # パラメータを調整せずに学習
    plt.figure(figsize=[5,8])
    for i in [0,1]:
        if(i):
            gp.saitekika(x0,y0,10000) # パラメータを調整する
        plt.subplot(211+i)
        plt.plot(x0,y0,'. ')
        mu,std = gp.yosoku(x1)
        plt.plot(x1,y(x1),'--r')
        plt.plot(x1,mu,'g')
        plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g')
        plt.title('a=%.3f, s=%.3f, w=%.3f'%tuple(gp.kernel.param))
    plt.tight_layout()
    plt.show()