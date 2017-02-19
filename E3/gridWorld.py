import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import  savefig
import warnings
from proGenerator import *
warnings.filterwarnings("ignore")

def gridworld():
    # if true, save the figure as pdf
    saveFigures = False
    # data of the generated grid world.
    data = genGridWorld()
    grid_world = data[0]
    grid_list = data[1]
    # calculate the transition dynamics P(s_t+1| s_t, a_t)
    # probModel = proGenerator()

    # ax = showWorld(grid_world, 'Environment')
    # showTextState(grid_world, grid_list, ax)

    if saveFigures:
        savefig('gridworld.pdf')

    # Finite Horizon
    V, policy = ValIter(R=grid_world, discount=1, T=15,infHor=False)
    V = V[:,:,0]
    print V
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
    # show policy
    ax = showWorld(grid_world, 'Policy - Finite Horizon')
    showPolicy(policy[:,:,5], ax)
    plt.show()

    # Infinite Horizon
    # V, policy= ValIter(R=grid_world, discount=0.8, T=15, infHor=True)
    # print V
    # showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
    # ax = showWorld(grid_world, 'Policy - Infinite Horizon')
    # showPolicy(policy, ax)
    # plt.show()
    # if saveFigures:
    #     savefig('value_Inf_08.pdf')

    # policy = findPolicy();
    # ax = showWorld(grid_world, 'Policy - Infinite Horizon')
    # showPolicy(policy, ax)
    # if saveFigures:
    #     savefig('policy_Inf_08.pdf')
    #
    # # Finite Horizon with Probabilistic Transition
    # V = ValIter()
    #
    # V = V[:,:,0];
    # showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
    # if saveFigures:
    #     savefig('value_Fin_15_prob.pdf')
    #
    # policy = findPolicy()
    # ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
    # showPolicy(policy, ax)
    # if saveFigures:
    #     savefig('policy_Fin_15_prob.pdf')


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35    # Dirt
    W = -100  # Water
    C = -3000 # Cat
    T = 1000  # Toy
    grid_list = {0:'', O:'O', D:'D', W:'W', C:'C', T:'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5,10.5,1))
    ax.set_yticks(np.arange(0.5,9.5,1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in xrange(grid_world.shape[0]):
        for y in xrange(grid_world.shape[1]):
            if grid_world[x,y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x,y]), xy=(y,x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in xrange(policy.shape[0]):
        for y in xrange(policy.shape[1]):
            if policy[x,y] == 0:
                ax.annotate('$\downarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 2:
                ax.annotate('$\uparrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 3:
                ax.annotate('$\leftarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 4:
                ax.annotate('$\perp$', xy=(y,x), horizontalalignment='center')


##
def ValIter(R, discount, T, infHor, probModel=np.array([])):
    row = np.size(R, 0)
    column = np.size(R, 1)
    if infHor == False:
        J = np.zeros(shape=(row, column, T))
        J[:,:,T-1] = R
        policy = np.zeros(shape=(row, column, T))
        policy[:,:,T-1] = 4 # action = stay
        # iteration time
        for t in range(T-2, -1, -1):
            J[:,:,t] = J[:,:,t+1]
            for row, rewards in enumerate(R):
                for column, reward in enumerate(rewards):
                    state = np.array([row, column])
                    rs = np.zeros(shape=5)
                    res = np.zeros(shape=(5,4))
                    for action in range(5):
                        # # problem a) finite horizon problem
                        nextState = next_state(state, action)
                        if nextState == []:
                            rs[action] = -1
                        else:
                            rs[action] = J[nextState[0],nextState[1],t+1]
                        # problem d) finite horizon problem with probabilistic transition function
                        # nextStates, prob = next_states(state, action)
                        # res[action,:] = myrewards(nextStates,J[:,:,t+1])
                        # summ = 0
                        # for xxx in range(4):
                        #     summ = summ + prob[xxx] * res[action, xxx]
                        # rs[action] = summ
                    r = np.max(rs)
                    policy[row,column,t] = np.argmax(rs)
                    J[row, column, t] = reward + r

    elif infHor == True:
        J = np.zeros(shape=(row,column))
        policy = np.zeros(shape=(row, column))
        t = 0
        while True:
            j = J.copy()
            d = 0
            for row, rewards in enumerate(R):
                for column, reward in enumerate(rewards):
                    state = np.array([row, column])
                    rs = np.zeros(shape=5)
                    for action in range(5):
                        nextStates = next_state(state, action)
                        rs[action] = j[nextStates[0],nextStates[1]]
                    r = np.max(rs)
                    J[row, column] = reward + discount * r
                    policy[row,column] = np.argmax(rs)
                    diff = np.fabs(J[row, column] - j[row, column])
                    print diff
                    if diff > d:
                        d = diff
            if d <= 1: break
            # if t > 20: break
    return J, policy
##
def maxAction(V, R, discount, probModel=np.array([])):
    """

    :param V: Value function
    :param R: Reward function
    :param discount: Discount factor, [0~1]
    :param probModel:
    :return:
    """
    pass

##
def findPolicy(V, probModel=np.array([])):
    """
    To find the optimal policy, a policy pi is a dictribution over actions given states.
    :param V: Value function
    :param probModel:
    :return:
    """
    row = np.size(V,0)
    column = np.size(V,1)
    for i in range(row):
        for j in range(column):
            state = np.array([row, column])
            print state

def main():
    gridworld()

if __name__ == "__main__":
    main()