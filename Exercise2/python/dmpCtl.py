# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np
def dmpCtl (dmpParams, psi_i, q, qd):
    # parameters
    a = dmpParams.alpha
    b = dmpParams.beta
    g = dmpParams.goal
    t = dmpParams.tau
    w = dmpParams.w
    # forcing function
    ft = np.matmul(np.transpose(psi_i), w)
    # calculate the \ddot(y)
    qdd = b * (g - q) - qd/t
    qdd = a * qdd
    qdd = np.square(t) * (qdd + ft)
    return qdd
