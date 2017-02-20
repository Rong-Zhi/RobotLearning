from getDMPBasis import *
from dmpCtl import *

def simSys ( states, dmpParams, dt, nSteps ):


    Phi = getDMPBasis (dmpParams, dt, nSteps)

    for i in range(nSteps - 1):
        # TO-DO: implement the dmpCtl() function, to get the y_dd(same symbol as the exercise)
        # y_dd = t^2( a(b(g-y)-y_d/t) + fw(z))
        qdd = dmpCtl( dmpParams, Phi[i,:].transpose(), states[i,::2], states[i,1::2].transpose() )
        #raw_input()
        states[i + 1,1::2] = states[i,1::2] + dt * qdd
        states[i + 1,::2] = states[i,::2] + dt * states[i,1::2]

    return states
