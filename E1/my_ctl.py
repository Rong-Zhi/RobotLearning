# CTL is the name of the controller.

# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints.

# Q and QD are the current position and velocity, respectively.

# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.

# GRAVITY is the gravity vector g(q).

# CORIOLIS is the Coriolis force vector c(q, qd).

# M is the mass matrix M(q).

import numpy as np


def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M):

    KP = np.array([60, 30])*10
    KD = np.array([10, 6])*10
    KI = np.array([0.1,0.1])*10
    error = q_des - q
    errord = qd_des - qd

    if ctl == 'P':
        u = np.zeros((2, 1))  # Implement your controller here
        print(u)
        u = np.multiply(KP,error)
        u = np.mat(u).transpose()
    elif ctl == 'PD':
        u = np.zeros((2, 1))  # Implement your controller here
        q_error = np.multiply(KP, error)
        qd_error = np.multiply(KD, errord)
        u = np.mat(q_error + qd_error).transpose()

    elif ctl == 'PID':
        u = np.zeros((2, 1))  # Implement your controller here
        q_error = np.multiply(KP, error)
        qd_error = np.multiply(KD, errord)
        qi_error = np.multiply(KI, np.sum(q_deshist - q_hist,0))
        u = np.mat(q_error + qd_error + qi_error).transpose()

    elif ctl == 'PD_Grav':
        u = np.zeros((2, 1))  # Implement your controller here
        q_error = np.multiply(KP, error)
        qd_error = np.multiply(KD, errord)
        u = np.mat(q_error + qd_error + gravity).transpose()

    elif ctl == 'ModelBased':
        u = np.zeros((2, 1))  # Implement your controller here
        qref = qdd_des + np.multiply(KP, error) + np.multiply(KD, errord)
        M = np.mat(M).transpose()
        u = np.dot(qref,M) + coriolis + gravity
        u = np.mat(u).transpose()
    return u
