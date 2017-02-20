# Launches the simulation for the DMP-based controller and plot the
# results.
#
# GOALS is an optional array of cells that specifies the desided positions
# and velocities of the joints.
#
# GOALS is an optional array of values of different taus for the DMP.
#
# FILENAME is the name of your output files. In the end the code will
# generate two pdf files named 'filename_q1.pdf' and
# 'filename_q2.pdf' containing the plots.

import numpy as np
from simSys import *
from DoubleLink import *
from getImitationData import *
from dmpTrain import *
from math import pi

def dmpComparison (goals, taus, filename):

    dt = 0.002

    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])


    t_end = 3.0

    sim_time = np.arange(dt, t_end-dt, dt)
    nSteps = len(sim_time)

    # Imitation Data
    data = getImitationData(dt, sim_time)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    # To train the robot with imitation data and get the parameters
    # fw(z) = phi(z)T * w;   dmpParams = w
    dmpParams = dmpTrain(q, qd, qdd, dt, len(sim_time))

    states = np.zeros((nSteps, 4))
    states[0,::2] = [-pi, 0]

    # plot 1, joint 0
    h1 = plt.figure()
    plt.hold('on')
    p1_1, = plt.plot(sim_time, q[0,:], linewidth=2.0, label='Desired $q_1$')

    h2 = plt.figure()
    plt.hold('on')
    # joint 1
    p2_1, = plt.plot(sim_time, q[1,:], linewidth=2.0, label='Desired $q_2$')

    states = simSys ( states, dmpParams, dt, nSteps )

    plt.figure(h1.number)
    p1_2, =plt.plot(sim_time, states[:,0], ':', color='r', linewidth=4.0, label='DMP $q_1$')
    plt.legend(loc=0)
    plt.figure(h2.number)
    p2_2, =plt.plot(sim_time, states[:,2], ':', color='r', linewidth=4.0, label='DMP $q_2$')
    plt.legend(loc=0)

    dmpParamsOld = dmpParams

    p1_h = [ 0, 0]
    p2_h = [ 0, 0]



    if goals != []:
        for i in range(len(goals)):
            states = np.zeros((nSteps, 4))
            states[0,::2] = [-pi, 0]
            dmpParams.goal = goals[i]
            states = simSys ( states, dmpParams, dt, nSteps )

            plt.figure(h1.number)
            p1_h[i], = plt.plot(sim_time,states[:,0],linewidth=2.0, label='DMP $q_1$ with goal = [' + str(goals[i][0]) + ']')
            plt.plot(sim_time[-1],goals[i][0],'kx',markersize = 15.0)

            plt.figure(h2.number)
            p2_h[i], = plt.plot(sim_time,states[:,2],linewidth=2.0, label='DMP $q_2$ with goal = [' + str(goals[i][1]) + ']')
            plt.plot(sim_time[-1],goals[i][1],'kx',markersize = 15.0)

        plt.figure(h1.number)
        # plt.legend([p1_1, p1_2, p1_h[0], p1_h[1]], loc= 0)
        plt.legend(loc=0)
        plt.figure(h2.number)
        # plt.legend([p2_1, p2_2, p2_h[0], p2_h[1]], loc= 0)
        plt.legend(loc=0)
    dmpParams = dmpParamsOld

    if taus != []:
        for i in range(len(taus)):
            states = np.zeros((nSteps, 4))
            states[0,::2] = [-pi, 0]
            dmpParams.tau = taus[i]
            states = simSys ( states, dmpParams, dt, nSteps )

            plt.figure(h1.number)
            p1_h[i], = plt.plot(sim_time,states[:,0],linewidth=2.0, label='DMP $q_1$ with $\tau$ = [' + str(taus[i]) + ']')

            plt.figure(h2.number)
            p2_h[i], = plt.plot(sim_time,states[:,2],linewidth=2.0, label='DMP $q_2$ with $\tau$ = [' + str(taus[i]) + ']')

        plt.figure(h1.number)
        # plt.legend(handles=[p1_1, p1_2, p1_h[0], p1_h[1]], loc= 0)
        plt.legend(loc=0)
        plt.figure(h2.number)
        # plt.legend(handles=[p2_1, p2_2, p2_h[0], p2_h[1]], loc= 0)
        plt.legend(loc=0)

    plt.pause(40)
