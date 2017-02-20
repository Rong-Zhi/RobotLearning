# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total. number of steps

from getDMPBasis import *
import numpy as np

class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = [0, 0.2]
        self.w = 0.0

def dmpTrain (q, qd, qdd, dt, nSteps):
    # shape of q, qd, qdd is (2, 1499)
    # nSteps = 1499
    params = dmpParams()
    #Set dynamic system parameters
    params.alphaz = 3/ (nSteps * dt)
    params.alpha  = 25
    params.beta	 = 6.25
    # params.Ts     = nSteps * dt
    params.tau    = 1
    params.nBasis = 50

    # Phi is the basis functions phi(z)
    Phi = getDMPBasis(params, dt, nSteps)


    # Parameters need to compute forcing function.
    a = params.alpha
    b = params.beta
    t = params.tau
    g = params.goal

    # Compute the forcing function
    ft = np.zeros(shape=(1499, 2))
    for i in range(nSteps):
        ft[i,:] = qdd[:,i]/np.square(t) - a * (b*(g-q[:,i])-qd[:,i]/t)

    #Learn the weights
    params.w = np.matmul(np.linalg.pinv(Phi), ft)

    return params
