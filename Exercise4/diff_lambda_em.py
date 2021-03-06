from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats
# %matplotlib inline

# Calculate Reward of each sample
def Reward_Cal(mu,sigma,num,dim):
    sample = np.zeros(shape=(num,dim))
    Reward = np.zeros(num)
    for i in range(num):
        sample[i,:] = np.random.multivariate_normal(mu,sigma)
        Reward[i] = env.getReward(sample[i,:])
    return sample,Reward

#Calculate Confidence
def Confidence_interval(R,confidence):
    mean = np.mean(R,0)
    std = scipy.stats.sem(R,0)
    down = mean - 1.96 * std
    up = mean + 1.96 * std
    # down,up = scipy.stats.t.interval(confidence,np.size(R,0)-1,loc=mean,scale=std)
    return mean, up, down

# Update Mu and Sigma
def meannvar(sample,Reward,l,num,dim):
    beta = l/(max(Reward)-min(Reward))
    meansum = np.zeros(dim)
    s_sum = np.zeros(shape=(dim,dim))
    weight = np.zeros(shape=(num,1))
    # tmp = np.zeros(shape=(1,dim))
    for i in range(num):
        weight[i] = np.exp((Reward[i]-max(Reward))*beta)
        meansum = meansum + weight[i] *sample[i,:]
    Mu_w = meansum/sum(weight)
    for i in range(num):
        tmp = sample[i,:] - Mu_w
        tmp = weight[i] * np.square(tmp)
        for m in range(dim):
            s_sum[m,m] = s_sum[m,m] + tmp[m]
        # s_sum = s_sum + weight[i] * (tmp.transpose() * tmp)
    newReward = np.mean(Reward)
    Sigma_w = s_sum/sum(weight)
    return Mu_w,Sigma_w,newReward



env = Pend2dBallThrowDMP()
numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10
delta_R = 10



l=25
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6
    sample = np.random.multivariate_normal(Mu_w, Sigma_w)
    Reward = env.getReward(sample)
    R = []
    R.append(Reward)
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        Mu_w,Sigma_w,meanRewards = meannvar(sample,Rewards,l,numSamples,numDim)
        R.append(meanRewards)
    Total_R[i,:] = R[1:maxIter+1]
mean3,up3,down3 = Confidence_interval(Total_R,confidence=0.95)

# ... then draw a sample and simulate an episode
l = 3
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6
    sample = np.random.multivariate_normal(Mu_w, Sigma_w)
    Reward = env.getReward(sample)
    R = []
    R.append(Reward)
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        Mu_w,Sigma_w,meanRewards = meannvar(sample,Rewards,l,numSamples,numDim)
        R.append(meanRewards)
    Total_R[i,:] = R[1:maxIter+1]

mean1,up1,down1 = Confidence_interval(Total_R,confidence=0.95)

l=7
Total_R = np.zeros(shape=(10,maxIter))
for i in range(10):
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6
    sample = np.random.multivariate_normal(Mu_w, Sigma_w)
    Reward = env.getReward(sample)
    R = []
    R.append(Reward)
    for t in range(maxIter):
        sample,Rewards = Reward_Cal(Mu_w,Sigma_w,numSamples,numDim)
        Mu_w,Sigma_w,meanRewards = meannvar(sample,Rewards,l,numSamples,numDim)
        R.append(meanRewards)
    Total_R[i,:] = R[1:maxIter+1]
mean2,up2,down2 = Confidence_interval(Total_R,confidence=0.95)



plt.figure()

plt.plot(mean1, 'b',label=r'$\lambda$=3')
plt.fill_between(np.arange(0, maxIter, 1),up1, down1,color='skyblue',alpha=0.5)
plt.yscale('symlog')


plt.plot(mean2, 'r',label=r'$\lambda$=7')
plt.fill_between(np.arange(0, maxIter, 1),up2, down2,color='tomato',alpha=0.5)
plt.yscale('symlog')

plt.plot(mean3, 'g',label=r'$\lambda$=25')
plt.fill_between(np.arange(0, maxIter, 1),up3, down3,color='lightgreen',alpha=0.5)
plt.yscale('symlog')


plt.xlabel('Time step')
plt.ylabel('Averaged Reward')
plt.title(r'Mean of the average return with different $\lambda$')
plt.legend(shadow=True)
plt.show()


