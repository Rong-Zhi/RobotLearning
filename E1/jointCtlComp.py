# This is one of the two main classes you need to run.
#
# CTLS is a list of cells with your controller name
# ('P','PD,'PID','PD_Grav','ModelBased'). You can run more controllers
# one after another, e.g. passing {'P','PD_Grav'}.
#
# ISSETPOINT is a boolean variable. If 1 the robot has to reach a fixed
# point, if 0 it will follow a fixed trajectory.
#
# PAUSETIME is the number of seconds between each iteration for the
# animated plot. If 0 only the final position of the robot will be
# displayed.
#
# FILENAME is the name of your output files. At the end the code will
# generate two pdf files named 'filename_position.pdf' and
# 'filename_velocity.pdf' containing the plots.

import numpy as np
from math import pi, sin, cos
from simSys import *
from DoubleLink import *


def jointCtlComp(ctls, isSetPoint, pauseTime, filename):
    if 'ctls' not in locals() or not ctls:
        ctls = ['P']

    if 'isSetPoint' not in locals() or not isSetPoint:
        isSetPoint = False

    if 'pauseTime' not in locals() or not pauseTime:
        pauseTime = False

    dt = 0.002
    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])

    t_end = 3.0

    time = np.arange(0, t_end, dt)

    nSteps = len(time)
    numContrlComp = len(ctls)

    target = {'cartCtl': False}
    if isSetPoint:
        target['q'] = np.tile([-pi/2, 0], (nSteps, 1))
        target['qd'] = np.tile([0, 0], (nSteps, 1))
        target['qdd'] = np.tile([0, 0], (nSteps, 1))
    else:
        f1 = 2
        f2 = 0.5
        target['q'] = np.array([np.sin(2 * pi * f1 * time) - pi, np.sin(2 * pi * f2 * time)]).transpose()
        target['qd'] = np.array(
            [2 * pi * f1 * np.cos(2 * pi * f1 * time), 2 * pi * f2 * np.cos(2 * pi * f2 * time)]).transpose()
        target['qdd'] = np.array([-(2 * pi * f1) ** 2 * np.sin(2 * pi * f1 * time),
                                  -(2 * pi * f2) ** 2 * np.sin(2 * pi * f2 * time)]).transpose()

    states = simSys(robot, dt, nSteps, ctls, target, pauseTime)
    colors = [1, 2, 3]

    traj_plot(states, colors, numContrlComp, ctls, target['q'], target['qd'], time, 0)

    traj_plot(states, colors, numContrlComp, ctls, target['q'], target['qd'], time, 1)

    plt.pause(5)

# Just a way to plot, feel free to modify!
def traj_plot(states, colors, numContrlComp, ctls, q_desired, qd_desired, time, plotVel):
    if plotVel:
        y = qd_desired
    else:
        y = q_desired

    plt.figure()
    plt.hold(True)

    h = plt.plot(time, y, linewidth=2)
    plt.draw()
    plt.setp(h[0], 'LineStyle', '-.')
    plt.setp(h[1], 'LineStyle', '-.')
    names = ['Desired_1', 'Desired_2']
    #
    for k in range(numContrlComp):
        names += [ctls[k] + '_1', ctls[k] + '_2']
        plt.plot(time, states[:, plotVel::2], linewidth=2)

    plt.legend(tuple(names))
    plt.xlabel('time(s)', fontsize=15)
    plt.ylabel('angle (rad)', fontsize=15)

    if plotVel:
        plt.title('Velocity', fontsize=20)
    else:
        plt.title('Position', fontsize=20)
