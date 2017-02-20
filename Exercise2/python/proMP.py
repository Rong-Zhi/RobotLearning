import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *
from scipy.linalg import block_diag
def proMP (nBasis):

    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time);
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    # shape (Phi) = (30, 1499)
    Phi = getProMPBasis( dt, nSteps, nBasis, bandwidth )

    # y = [q, qd]
    w = np.matmul(np.linalg.pinv(Phi.transpose()),q.transpose())
    mean_w = np.mean(w,axis=1)
    cov_w = np.cov(w)
    # print np.std(w, axis=1)
    # print np.cov(w)

    # shape(w) = (30, 45)
    # Phi = np.transpose(Phi[0:nSteps])
    # plt.figure()
    # plt.hold('on')
    # plt.fill_between(time, np.dot(Phi.transpose(),mean_w) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w,Phi)))), np.dot(Phi.transpose(),mean_w) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w,Phi)))), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    # plt.plot(time,np.dot(Phi.transpose(),mean_w), color='#1B2ACC')
    # plt.plot(time,q.transpose())
    # plt.title('ProMP learned from several trajectories(N=20)')
    # plt.title('Trajectories used for imitation.')
    # plt.pause(10)

    #Conditioning
    y_d = 3
    Sig_d = 0.0002
    t_point = np.round(2300/2)

    Phi_t_T = np.array(Phi[:, t_point])[np.newaxis]
    Phi_t = np.transpose(Phi_t_T)
    covw_phit = np.dot(cov_w, Phi_t)
    x1 = 1 / ( Sig_d + np.dot(Phi_t_T, covw_phit))
    mean_w_new = np.dot(covw_phit, np.dot(x1, y_d-np.dot(Phi_t_T, mean_w)))
    mean_w_new = np.asarray(mean_w + mean_w_new)
    cov_w_new = np.dot(covw_phit, np.dot(x1, np.dot(Phi_t_T, cov_w)))
    cov_w_new = np.asarray(cov_w - cov_w_new)

    plt.figure()
    plt.hold('on')
    # plt.fill_between(time, np.dot(Phi.transpose(),mean_w) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w,Phi)))), np.dot(Phi.transpose(),mean_w) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w,Phi)))), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    # plt.plot(time,np.dot(Phi.transpose(),mean_w), color='#1B2ACC')
    # plt.fill_between(time, np.dot(Phi.transpose(),mean_w_new) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w_new,Phi)))), np.dot(Phi.transpose(),mean_w_new) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(),np.dot(cov_w_new,Phi)))), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # plt.plot(time,np.dot(Phi.transpose(),mean_w_new), color='#CC4F1B')
    sample_traj = np.dot(Phi.transpose(),np.random.multivariate_normal(mean_w_new,cov_w_new,10).transpose())
    plt.plot(time,sample_traj)
    # plt.title('ProMP after contidioning with new sampled trajectories')
    plt.title('New computed trajectories from sampled K=10 random weights vectors')
    plt.pause(20)
