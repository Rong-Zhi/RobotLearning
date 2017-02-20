import numpy as np

def myrewards(states,grid_world):
    Nstates = np.size(states,1)
    rs = np.zeros(shape=(4))
    for i in range(Nstates):# 0~3
        # print states[:,i], probs[i]
        # if states[:,i]
        x = states[1,i]
        y = states[0,i]
        if x >=0 and x <= 9 and y >=0 and y <= 8:
            rs[i] = grid_world[y,x]
        else:
            rs[i] = -1000000
    return rs

def next_states(state,action):
    nextStates = np.zeros(shape=(2,4),dtype='int32')
    prob = np.zeros(shape=(4))
    top = 0
    bottom = 8
    left = 0
    right = 9
    x = state[1]
    y = state[0]
    if action == 0: # down
        if y != bottom:
            nextStates[0,0] = y + 1 # go down
            nextStates[1,0] = x
            prob[0] = 0.7
            nextStates[0,1] = y # go right
            nextStates[1,1] = x + 1
            prob[1] = 0.1
            nextStates[0,2] = y # go left
            nextStates[1,2] = x - 1
            prob[2] = 0.1
            nextStates[0,3] = y # stay
            nextStates[1,3] = x
            prob[3] = 0.1
            # prob = np.array([0.9, 0.1, 0, 0])
    elif action == 1: # right
        if x != right:
            nextStates[0, 0] = y  # go right
            nextStates[1, 0] = x + 1
            prob[0] = 0.7
            nextStates[0, 1] = y - 1  # up
            nextStates[1, 1] = x
            prob[1] = 0.1
            nextStates[0, 2] = y + 1  # down
            nextStates[1, 2] = x
            prob[2] = 0.1
            nextStates[0, 3] = y  # stay
            nextStates[1, 3] = x
            prob[3] = 0.1
            # prob = np.array([0.9, 0.1, 0, 0])
    elif action == 2: # up
        if y != top:
            nextStates[0, 0] = y - 1  # go up
            nextStates[1, 0] = x
            prob[0] = 0.7
            nextStates[0, 1] = y     # right
            nextStates[1, 1] = x + 1
            prob[1] = 0.1
            nextStates[0, 2] = y    # left
            nextStates[1, 2] = x -1
            prob[2] = 0.1
            nextStates[0, 3] = y  # stay
            nextStates[1, 3] = x
            prob[3] = 0.1
            # prob = np.array([0.9, 0.1, 0, 0])
    elif action == 3: # left
        if x != left:
            nextStates[0, 0] = y   #  go left
            nextStates[1, 0] = x - 1
            prob[0] = 0.7
            nextStates[0, 1] = y - 1  # up
            nextStates[1, 1] = x
            prob[1] = 0.1
            nextStates[0, 2] = y + 1  # down
            nextStates[1, 2] = x
            prob[2] = 0.1
            nextStates[0, 3] = y  # stay
            nextStates[1, 3] = x
            prob[3] = 0.1
            # prob = np.array([0.9, 0.1, 0, 0])
    elif action == 4:# stay
        nextStates[0,:] = y
        nextStates[1,:] = x
        prob[:] = 0.25
    return nextStates, prob


def next_state(state, action):
    nextState = np.zeros(shape=2,dtype='int32')
    top = 0
    bottom = 8
    left = 0
    right = 9
    x = state[1]
    y = state[0]
    if action == 0:  # down
        if y == bottom:
            nextState = []
        else:
            nextState[0] = y + 1
            nextState[1] = x
    elif action == 1:  # right
        if x == right:
            nextState = []
        else:
            nextState[0] = y
            nextState[1] = x + 1
    elif action == 2:  # up
        if y == top:
            nextState = []
        else:
            nextState[0] = y - 1
            nextState[1] = x
    elif action == 3:  # left
        if x == left:
            nextState = []
        else:
            nextState[0] = y
            nextState[1] = x - 1
    elif action == 4:  # stay
        nextState[0] = y
        nextState[1] = x
    return nextState
