import numpy as np


At = np.matrix('1 0.1; 0 1')
Bt = np.matrix('0;0.1')
#
bt = np.matrix('5;0')
Sigma = 0.01
#
Kt = np.matrix('5 0.3')

kt = 0.3
#
Ht = 1
Rt1 = np.matrix('100000 0; 0 0.1')
Rt2 = np.matrix('0.01 0; 0 0.1')
rt1 = np.matrix('10;0')
rt2 = np.matrix('20;0')


