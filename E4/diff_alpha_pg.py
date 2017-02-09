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

#Calculate gradient
def mu_gradient(mu,sigma,num,dim,theta,Rewards):
    d_mu = np.zeros(dim)
    for i in range(num):
        tmp = (Rewards[i]-np.mean(Rewards))
        # tmp = Rewards[i]
        d_mu = d_mu + tmp * (theta[i,:]-mu[:])/np.diag(sigma) **2
    return d_mu


env = Pend2dBallThrowDMP()
numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

alpha = 0.1
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 10
    R = []
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        d_mu = mu_gradient(Mu_w,Sigma_w,numSamples,numDim,sample,Rewards)
        Mu_w = Mu_w + alpha * d_mu
        R.append(np.mean(Rewards))
    Total_R[i,:] = R[:]
#When alpha=0.1
mean1,up1,down1 = Confidence_interval(Total_R,confidence=0.95)

#when alpha=0.2
alpha = 0.2
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 10
    R = []
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        d_mu = mu_gradient(Mu_w,Sigma_w,numSamples,numDim,sample,Rewards)
        Mu_w = Mu_w + alpha * d_mu
        R.append(np.mean(Rewards))
    Total_R[i,:] = R[:]
#
mean2,up2,down2 = Confidence_interval(Total_R,confidence=0.95)

#when alpha=0.4
alpha = 0.4
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 10
    R = []
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        d_mu = mu_gradient(Mu_w,Sigma_w,numSamples,numDim,sample,Rewards)
        Mu_w = Mu_w + alpha * d_mu
        R.append(np.mean(Rewards))
    Total_R[i,:] = R[:]
#
mean3,up3,down3 = Confidence_interval(Total_R,confidence=0.95)



plt.figure()

plt.plot(mean1, 'b',label=r'$\alpha$=0.1')
plt.fill_between(np.arange(0, maxIter, 1),up1, down1,color='skyblue')
plt.yscale('symlog')


plt.plot(mean2, 'r',label=r'$\alpha$=0.2')
plt.fill_between(np.arange(0, maxIter, 1),up2, down2,color='tomato')
plt.yscale('symlog')

plt.plot(mean3, 'g',label=r'$\alpha$=0.4')
plt.fill_between(np.arange(0, maxIter, 1),up3, down3,color='lightgreen')
plt.yscale('symlog')


plt.xlabel('Time step')
plt.ylabel('Averaged Reward')
plt.title(r'Mean of the average return with different $\lambda$')
plt.legend(shadow=True)
plt.show()

