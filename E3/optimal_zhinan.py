import numpy as np
import scipy as sp
import scipy.stats
from matplotlib import pyplot as plt

# Initialize all needed parameters
T = 50
At = np.matrix('1 0.1; 0 1')
Bt = np.matrix('0;1')
bt = np.matrix('5;0')
Sigma_t = 0.01
Kt = np.matrix('5 0.3')
kt = 0.3
Ht = 1

Rt1 = np.matrix('100000 0; 0 0.1')
Rt2 = np.matrix('0.01 0; 0 0.1')
rt1 = np.matrix('10;0')
rt2 = np.matrix('20;0')


# function to compute mean and confidence interval
def mean_confidence_interval(data, confidence=0.95):
    m, se = np.mean(data, 0), scipy.stats.sem(data, 0)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# function to compute the states and control signals
def states_and_signal(T, Kt, s0, kt, bt, Sigma_t,
                      At, Bt, rt1, rt2, Rt1, Rt2, Ht):
    # dummy vector to store the control signal at
    at = np.zeros([T + 1, 1])
    # dummy matrix to store the states st
    st = np.zeros([2, T + 1])
    st[:, 0] = np.transpose(s0)
    cum_reward = 0

    for t in range(0, T + 1, 1):
        if t <= 14:
            #at[t] = np.dot(Kt, (rt1-np.matrix(st[:,t]).transpose())) + kt
            at[t] = np.dot(Kt, (0 - np.matrix(st[:, t]).transpose())) + kt
        else:
            #at[t] = np.dot(Kt, (rt2 - np.matrix(st[:,t]).transpose())) + kt
            at[t] = np.dot(Kt, (0 - np.matrix(st[:, t]).transpose())) + kt

        if 40 < t <= T or 14 < t < 40:
            Rt = Rt2
            rt = rt2
        elif t == 40:
            Rt = Rt2
            rt = rt2
        elif t == 14:
            Rt = Rt1
            rt = rt1
        else:
            Rt = Rt2
            rt = rt1

        wt = np.random.normal(bt, Sigma_t, (2, 1))
        st_temp = np.dot(At, s0) + np.multiply(Bt, at[t]) + wt

        if t == T:
            st_temp = np.dot(At, s0) + np.multiply(Bt, at[t]) + wt
            st_now = np.matrix(st[:, t]).transpose()
            reward = -np.dot(np.dot(np.transpose(st_now - rt), Rt), st_now - rt)
        else:
            st_temp = np.dot(At, s0) + np.multiply(Bt, at[t]) + wt
            st_now = np.matrix(st[:, t]).transpose()
            st[:, t + 1] = np.transpose(st_temp)
            reward = -np.dot(np.dot(np.transpose(st_now - rt), Rt), st_now
                             - rt) - np.dot(Ht * np.transpose(at[t]),at[t])
        cum_reward = cum_reward + reward

    return at, st, cum_reward


# dummy vector to store control signals and states into matrices
at = np.zeros([1, T + 1])
st1 = at
st2 = st1
cum_reward = np.zeros([1, 20])
# run the system 20 times and store the states
# and control signals for each realization
for n in range(0, 20, 1):
    s0 = np.random.normal(0, 1, (2, 1))
    at_temp, st_temp, cum_reward[:, n] = states_and_signal(T, Kt, s0, kt, bt,
    Sigma_t, At, Bt, rt1, rt2, Rt1, Rt2, Ht)
    at = np.vstack((at, np.transpose(at_temp)))
    st1 = np.vstack((st1, st_temp[1, :]))
    st2 = np.vstack((st2, st_temp[0, :]))

# delte the first zero columns
at = at[1:-1, :]
print at
st1 = st1[1:-1, :]
st2 = st2[1:-1, :]

# compute the corresponding means and confidence intervals
m_at, low_at, up_at = mean_confidence_interval(at)
m_st1, low_st1, up_st1 = mean_confidence_interval(st1)
m_st2, low_st2, up_st2 = mean_confidence_interval(st2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

ax1.plot(m_at, 'r')
ax1.fill_between(np.arange(0, 51, 1), low_at, up_at)
ax1.set_ylabel('Control Signal')

ax2.plot(m_st1, 'r')
ax2.fill_between(np.arange(0, 51, 1), low_st1, up_st1)
ax2.set_ylabel('State 1')

ax3.plot(m_st2, 'r')
ax3.fill_between(np.arange(0, 51, 1), low_st2, up_st2)
ax3.set_ylabel('State 2')
ax3.set_xlabel('t')

print 'The mean of the cumulative reward is: ', np.mean(cum_reward)
print 'The standard deviation of the cumulative reward is: ', np.std(cum_reward)

plt.show()
