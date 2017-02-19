#Generates the trajectory we want to imitate.
import numpy as np
import scipy.interpolate as interp
from math import pi

def getImitationData (dt, time, multiple_demos=False):

    if multiple_demos:
        np.random.seed(10)
        nDemo = 45
        f = np.random.randn(nDemo,1) * 0.05 + 2
        aux = np.zeros((f.shape[0], len(time[500:1001])))
        for i in range(aux.shape[0]):
            aux[i,:] = f[i]*time[500:1001]
        qs = np.sin(2 * pi * aux)

        x = [0, dt] + list(time[500:1001]) + [time[-2], time[-1]]

        q = []
        for i in range(nDemo):
            y = [-pi, -pi] + list(qs[i,:]) + [0.3, 0.3]

            f1 = interp.interp1d(x, y, 'cubic')
            q1 = f1(time)
            if q == []:
                q = (np.random.randn(1) * 0.35+1) + q1
            else:
                q = np.vstack([q,(np.random.randn(1) * 0.35+1) + q1])
    else:

        qs = np.sin(2 * pi * 2 * time[500:1001])
        x = [0, dt] +  list(time[500:1001]) + [time[-2], time[-1]]
        y = [-pi, -pi] + list(qs) + [0.3, 0.3]

        f1 = interp.interp1d(x, y, 'cubic')
        q1 = f1(time)

        f2 = interp.interp1d([0,dt,time[-2],time[-1]],[0, 0, -0.8, -0.8],'cubic')
        q2 = f2(time)

        q  = np.array([list(q1), list(q2)])

    qd = np.diff(q)/dt
    aux = np.array(qd[:,-1], ndmin=2).transpose()
    qd = np.hstack([qd,aux])
    qdd = np.diff(qd)/dt
    aux = np.array(qdd[:,-1], ndmin=2).transpose()
    qdd = np.hstack([qdd,aux])

    return [q, qd, qdd]
