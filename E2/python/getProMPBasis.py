import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):
    time = np.arange(dt,nSteps*dt,dt)
    Ts = nSteps * dt - dt
    Phi   = np.zeros(shape=(n_of_basis,nSteps))
    C = np.zeros(shape=n_of_basis)
    for i in range(n_of_basis):
        C[i] = (Ts+4*bandwidth)*i/n_of_basis - 2*bandwidth
    for x in range(nSteps):
        for n in range(n_of_basis):
            Phi[n, x] = np.exp(-np.square(time[x]-C[n])/(2*np.square(bandwidth)))
        Phi[:, x] = Phi[:, x] / np.sum(Phi[:, x])
    plt.figure()
    plt.plot(time, Phi.transpose())
    plt.title('Basis Functions')
    return Phi

