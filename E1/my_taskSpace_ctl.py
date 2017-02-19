# CTL is the name of the controller.

# DT is the time step.

# Q and QD are respectively the position and the velocity of the joints.

# GRAVITY is the gravity vector g(q).

# CORIOLIS is the Coriolis force vector c(q, qd).

# M is the mass matrix M(q).

# J is the Jacobian.

# CART are the coordinates of the current position.

# DESCART are the coordinates of the desired position.

import numpy as np
from math import pi

def my_taskSpace_ctl(ctl, dt, q, qd, gravity, coriolis, M, J, cart, desCart):

    KP = np.diag([60, 30])
    KD = np.diag([10, 6])
    qrest = (0,pi)
    #qrest = (0,-pi)
    qrest = np.mat(qrest).transpose()
    gamma = 0.6
    dFact = 1e-6

    if ctl == 'JacTrans':
        qd_des = gamma * J.transpose() * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity

    elif ctl == 'JacPseudo':
        qd_des = gamma * J.transpose() * np.linalg.pinv(J * J.transpose()) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity

    elif ctl == 'JacDPseudo':
        qd_des = J.transpose() * np.linalg.pinv(J * J.transpose() + dFact * np.eye(2)) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity

    elif ctl == 'JacNullSpace':
        u = np.zeros((2, 1)) # Implement your controller here
        qd_0 = KP * (qrest - q)
        J_pseudo = J.transpose() * np.linalg.pinv(J * J.transpose()+dFact * np.eye(2))
        qd_des = J_pseudo * (desCart - cart) + (np.eye(2) - J_pseudo * J) *(qd_0)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP, KD])) * np.vstack([error, errord]) + coriolis + gravity
    return u