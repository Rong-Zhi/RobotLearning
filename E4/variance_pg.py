from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats


# Calculate Reward of each sample
def Reward_Cal(mu,sigma,num,dim):
    sample = np.zeros(shape=(num,dim))
    Reward = np.zeros(num)
    for i in range(num):
        sample[i,:] =  np.random.multivariate_normal(mu,sigma)
        Reward[i] = env.getReward(sample[i,:])
    return sample,Reward

#Calculate Confidence
def Confidence_interval(R,confidence):
    mean = np.mean(R,0)
    std = scipy.stats.sem(R,0)
    down,up = scipy.stats.t.interval(confidence,np.size(R,0)-1,loc=mean,scale=std)
    return mean, up, down

#Calculate gradient of mu and sigma
def gradient(mu,sigma,num,dim,theta,Rewards):
    d_mu = np.zeros(dim)
    d_sigma = np.zeros(shape=(dim,dim))
    for i in range(num):
        tmp = (Rewards[i]-np.mean(Rewards))
        # tmp = Rewards[i]
        d_mu = d_mu + tmp * (theta[i,:] - mu[:])/np.diag(sigma)**2
        d_sigma = d_sigma + tmp * ((theta[i] - mu[:])**2 - np.diag(sigma)**2)/np.diag(sigma)**3
    return d_mu,d_sigma



env = Pend2dBallThrowDMP()
numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10
alpha = 0.8
delta_R = 10
# Total_R = np.zeros(shape=(10,maxIter))
# for i in range(10):
Mu_w = np.zeros(numDim)
Sigma_w = np.eye(numDim) * 1e6
sample = np.random.multivariate_normal(Mu_w, Sigma_w)
Reward = env.getReward(sample)
R = []
R.append(Reward)
for t in range(maxIter):
    sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
    d_mu,d_sgma = gradient(Mu_w,Sigma_w,numSamples,numDim,sample,Rewards)
    Mu_w = Mu_w + alpha * d_mu

    Sigma_w = Sigma_w + alpha * d_sgma
    R.append(np.mean(Rewards))
    print("Mean Rewards:",np.mean(Rewards))
    print("Sigma:",Sigma_w.min())
    # delta_R = np.abs(R[t]-R[t-1])
    # if delta_R < 1e-3:
    #     Sigma_w_x = Sigma_w + np.eye(numDim)
    #     _,Rewards_x = Reward_Cal(Mu_w,Sigma_w_x,numSamples,numDim)
    #     meanRewards_x = np.mean(Rewards_x)
    #     t = t + 1
    #     dR = np.abs(R[t]-R[t-1])
    #     print("abs:",dR)
    #     if dR < 1e-3 :
    #         print("iteration time: ", t)
    #         break
#     Total_R[i,:] = R[:]
# #
# mean,up,down = Confidence_interval(Total_R,confidence=0.95)
# #
# plt.figure()
#
# plt.plot(mean, 'r')
# plt.xlabel('Time step')
# plt.ylabel('Averaged Reward')
#
# plt.plot(up,'b')
# plt.plot(down,'b')
# plt.fill_between(np.arange(0, maxIter, 1),up, down,color='skyblue')
# # plt.yscale('symlog')
# plt.show()
#
