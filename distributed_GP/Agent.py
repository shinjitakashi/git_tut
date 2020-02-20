import numpy as np

class Kernel:
    def __init__(self,param,bound=None):
        self.param = np.array(param)
        if(bound==None):
            bound = np.zeros([len(param),2])
            bound[:,1] = np.inf
        self.bound = np.array(bound)

    def __call__(self,x1,x2):
        a1,s,a2 = self.param
        return a1**2*np.exp(-0.5*((x1-x2)/s)**2) + a2**2*(x1==x2)

class Gausskatei:
    def __init__(self,kernel):
        self.kernel = kernel

    def gakushuu(self,x0,y0): # パラメータを調整せず学習
        self.x0 = x0
        self.y0 = y0
        self.k00 = self.kernel(*np.meshgrid(x0,x0))
        self.k00_1 = np.linalg.inv(self.k00)

    def yosoku(self,x): # xからyを予測
        k00_1 = self.k00_1
        k01 = self.kernel(*np.meshgrid(self.x0,x,indexing='ij'))
        k10 = k01.T
        k11 = self.kernel(*np.meshgrid(x,x))

        mu = k10.dot(k00_1.dot(self.y0))
        sigma = k11 - k10.dot(k00_1.dot(k01))
        std = np.sqrt(sigma.diagonal())
        return mu,std

    def logyuudo(self,param=None): # 対数尤度
        if(param is None):
            k00 = self.k00
            k00_1 = self.k00_1
        else:
            self.kernel.param = param
            k00 = self.kernel(*np.meshgrid(self.x0,self.x0))
            k00_1 = np.linalg.inv(k00)
        return -(np.linalg.slogdet(k00)[1]+self.y0.dot(k00_1.dot(self.y0)))

    def saitekika(self,xd,yd,kurikaeshi=1000): # パラメータを調整して学習
        self.xd = xd
        self.yd = yd
        param = self.kernel.param
        logbound = np.log(self.kernel.bound)
        s = (logbound[:,1]-logbound[:,0])/10.
        n_param = len(param)
        theta0 = np.log(param)
        p0 = self.logyuudo(param)
        lis_theta = []
        lis_p = []
        for _ in range(kurikaeshi):
            idou = np.random.normal(0,s,n_param)
            hazure = (theta0+idou<logbound[:,0])|(theta0+idou>logbound[:,1])
            while(np.any(hazure)):
                idou[hazure] = np.random.normal(0,s,n_param)[hazure]
                hazure = (theta0+idou<logbound[:,0])|(theta0+idou>logbound[:,1])
            theta1 = theta0 + idou
            param = np.exp(theta1)
            p1 = self.logyuudo(param)
            r = np.exp(p1-p0)
            if(r>=1 or r>np.random.rand()):
                theta0 = theta1
                p0 = p1
                lis_theta.append(theta0)
                lis_p.append(p0)
        self.ar_theta = np.array(lis_theta)
        self.ar_p = np.array(lis_p)
        self.kernel.param = np.exp(lis_theta[np.argmax(lis_p)])
        self.k00 = self.kernel(*np.meshgrid(self.x0,self.x0))
        self.k00_1 = np.linalg.inv(self.k00)