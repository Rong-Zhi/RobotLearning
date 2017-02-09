from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import scipy.stats
# %matplotlib inline



#Calculate Confidence
def Confidence_interval(R,confidence):
    mean = np.mean(R,0)
    std = scipy.stats.sem(R,0)
    down,up = scipy.stats.t.interval(confidence,np.size(R,0)-1,loc=mean,scale=std)
    return mean, up, down

# Calculate Reward of each sample
def Reward_Cal(mu,sigma,num,dim):
    sample = np.zeros(shape=(num,dim))
    Reward = np.zeros(num)
    for i in range(num):
        sample[i,:] = np.random.multivariate_normal(mu,sigma)
        Reward[i] = env.getReward(sample[i,:])
    return sample,Reward

# Update Mu and Sigma
def meannvar(sample,Reward,l,num,dim):
    beta = l/(max(Reward)-min(Reward))
    meansum = np.zeros(dim)
    s_sum = np.zeros(shape=(dim,dim))
    weight = np.zeros(num)
    for i in range(num):
        weight[i] = np.exp((Reward[i]-max(Reward))*beta)
        meansum = meansum + weight[i] *sample[i,:]
    Mu_w = meansum/sum(weight)
    for i in range(num):
        tmp = np.matrix(sample[i,:] - Mu_w[:])
        s_sum = s_sum + weight[i] * np.matmul(tmp.transpose(),tmp)
    newReward = np.mean(Reward)
    Sigma_w = s_sum / sum(weight)
    return Mu_w,Sigma_w,newReward


env = Pend2dBallThrowDMP()
numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10
l = 7


# ... then draw a sample and simulate an episode

delta_R = 10
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
    Total_R[i,:] = R[1:maxIter+1]
# print("Reward:",R[-1])

mean,up,down = Confidence_interval(Total_R,confidence=0.95)

plt.figure()

plt.plot(mean, 'b')
plt.xlabel('Time step')
plt.ylabel('Averaged Reward')
plt.title('Mean of the average return with 95% confidence')

plt.plot(up,'b')
plt.plot(down,'b')
plt.fill_between(np.arange(0, maxIter, 1),up, down,color='skyblue')
plt.yscale('symlog')
plt.show()


# Save animation
# env.animate_fig ( np.random.multivariate_normal(Mu_w,Sigma_w) )
# plt.savefig('EM-Ex2.pdf')

