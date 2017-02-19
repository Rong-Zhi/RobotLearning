import numpy as np
import constant as c
import matplotlib.pyplot as plt
# to calculate the states St and control signal at
class Optimal:
    def __init__(self):
        self.T = 50
        self.N = 20
    def stateANDaction(self):
        SN = []
        aN = []
        for n in range(self.N):
            S = []
            a = []
            s0 = np.random.normal(0, 1, (2, 1))
            s0_remove = s0
            S.append(s0)
            for t in range(1,self.T+1,1):
                wt = np.random.normal(c.bt, c.Sigma, (2, 1))
                at = 0 - np.dot(c.Kt, s0) + c.kt
                st_1 = np.dot(c.At, s0) + np.matmul(c.Bt, at) + wt
                s0 =  st_1
                a.append(at)
                S.append(s0)
            S.remove(s0_remove)
            SN.append(np.array(S))
            aN.append(np.array(a))
        SN = np.array(SN)
        aN = np.array(aN)
        # print np.shape(SN)
        SN_mean = np.zeros(shape=(self.T+1,2))
        aN_mean = np.zeros(shape=self.T+1)
        for t in range(self.T):
            SN_mean[t,0] = np.mean(SN[:,t,0,0])
            SN_mean[t,1] = np.mean(SN[:,t,1,0])
            aN_mean[t] = np.mean(SN[:,t,0,0])
        return SN, aN, SN_mean, aN_mean

    def reward(self,S, a):
        T = self.T
        N = self.N
        r = np.zeros(shape=(N,T))
        r_mean = np.zeros(shape=(self.T))
        for n in range(N):
            for t in range(T):
                if t <= 14:
                    rt = c.rt1
                elif t >=15:
                    rt = c.rt2
                s_r = np.matrix(S[n,t,:]).transpose() - rt
                s_rT = s_r.T
                if t == 50 or t == 100:
                    Rt = c.Rt1
                else:
                    Rt = c.Rt2
                # print np.shape(np.matmul(np.matmul(s_rT, Rt), s_r))
                r[n,t] = -np.matmul(np.matmul(s_rT, Rt), s_r)
                if t < T:
                    x =  a[n,t] * c.Ht
                    x = x * a[n,t]
                    r[n,t] = r[n,t] - x
        for t in range(T):
            r_mean[t] = np.mean(r[:,t])
        return r, r_mean

def main():
    o = Optimal()
    S, a, S_mean, a_mean = o.stateANDaction()
    r, r_mean = o.reward(S[:,:,:,0],a[:,:,0,0])
    print np.sum(r_mean)
    S = np.array(S)
    a = np.array(a)

    plt.figure()
    plt.plot(S_mean[:,0])
    plt.show()

    # plt.figure()
    # plt.plot(r_mean)
    # plt.show()

    # plt.figure()
    # plt.plot(S[:,:,1,0].transpose())
    # plt.figure()
    # plt.plot(a[:,:,0,0].transpose())
    # plt.show()

if __name__== "__main__":
    main()


    # def stateANDaction(self):
    #     # S = np.zeros(shape=(self.T,self.N,2))
    #     # S_mean = np.zeros(shape=(self.T,2))
    #     # a = np.zeros(shape=(self.T,self.N,1))
    #     # a_mean = np.zeros(shape=(self.T,1))
    #     # w = np.zeros(shape=(self.T,self.N,2))
    #     # for n in range(self.N):
    #     #     S[0,n,:] = np.random.normal(0,1,(2,1))
    #     #     for t in range(self.T-1):
    #     #         a[t,n,0] = - np.dot(c.Kt, S[t,n,:]) + c.kt
    #     #         w[t,n,0] = np.random.normal(c.bt[0], c.Sigma)
    #     #         w[t,n,1] = np.random.normal(c.bt[1], c.Sigma)
    #     #         S[t+1,n,:] = np.dot(c.At, S[t,n,:]) + np.multiply(c.Bt, a[t,n,0]) + w[t,n,:]
    #     # for t in range(self.T):
    #     #     S_mean[t,0] = np.mean(S[t,:,0])
    #     #     S_mean[t,1] = np.mean(S[t,:,1])
    #     #     a_mean[t,0] = np.mean(a[t,:,0])
    #     # return S, a, S_mean, a_mean