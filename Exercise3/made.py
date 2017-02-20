import numpy as np
from math import fabs

class SumDict(dict):
    def __setitem__(self, key, value):
        if self.has_key(key):
            value += self.get(key)
        dict.__setitem__(self, key, value)

class MDP():

    def __init__(self):
        self.reward = np.array([
            [-0.04, -0.04, -0.04,     1],
            [-0.04,  None, -0.04,    -1],
            [-0.04, -0.04, -0.04, -0.04],
        ])
        self.initial_state = (0, 0)
        self.finals = [(0,3), (1,3)]
        self.actions = ('l', 'r', 'u', 'd')

    def __iter__(self):
        class Iterator:
            def __init__(self, iterator, finals):
                self.iterator = iterator
                self.finals = finals
            def next(self):
                while True:
                    coords = self.iterator.coords
                    val = self.iterator.next()
                    if val and coords not in self.finals: break
                    #if val: break
                return coords, val
        return Iterator(self.reward.flat, self.finals)


    def _move(self, state, action):
        """calculates the next state given an action"""
        shape = self.reward.shape
        next = list(state)
        if state in self.finals:
            pass
        elif action == 'l' and \
            (state[1] > 0 and self.reward[state[0]][state[1]-1] != None):
            next[1] -= 1
        elif action == 'r' and \
            (state[1] < shape[1]-1 and self.reward[state[0]][state[1]+1] != None):
            next[1] += 1
        elif action == 'u' and \
            (state[0] > 0 and self.reward[state[0]-1][state[1]] != None):
            next[0] -= 1
        elif action == 'd' and \
            (state[0] < shape[0]-1 and self.reward[state[0]+1][state[1]] != None):
            next[0] += 1
        return tuple(next)

    def successors(self, state, action):
        """this function returns a dict with all the successors of
        a state and the relative probability given an action,
        for example if you are in (1,0) and you want to go right
        the function returns {(1,0):0.8, (0,0):0.1, (2,0):0.1}"""

        #I'm using SumDict because if two or more "_move"
        #return the same state (and the state is the key of the dict)
        #we need to sum out the values not overwrite the old value
        d = SumDict()
        if action == 'l':
            d[self._move(state, 'l')] = 0.8
            d[self._move(state, 'u')] = 0.1
            d[self._move(state, 'd')] = 0.1
        elif action == 'r':
            d[self._move(state, 'r')] = 0.8
            d[self._move(state, 'u')] = 0.1
            d[self._move(state, 'd')] = 0.1
        elif action == 'u':
            d[self._move(state, 'u')] = 0.8
            d[self._move(state, 'l')] = 0.1
            d[self._move(state, 'r')] = 0.1
        elif action == 'd':
            d[self._move(state, 'd')] = 0.8
            d[self._move(state, 'l')] = 0.1
            d[self._move(state, 'r')] = 0.1
        return d

    # from_state, to_state: a tuple
    # action: l, r, u, d (left, right, up, down)
    def transition(self, from_state, action, to_state):
        return self.successors(from_state, action)[to_state]

    def initial_state(self):
        return self.initial_state

    def reward(self, state):
        return self.reward[state[0]][state[1]]

    def discount(self):
        return 1

def value_iteration(mdp):
    _J = np.zeros(mdp.reward.shape)
    for final in mdp.finals:
        _J[final[0]][final[1]] = mdp.reward.item(final)
    while True:
        j = _J.copy()
        d = 0
        for state, reward in mdp:
            summ = max([sum([prob * j.item(next_state) for next_state, prob in
                mdp.successors(state, action).items()]) for action in
                mdp.actions])
            _J[state[0]][state[1]] = reward + mdp.discount() * summ
            diff = fabs(_J.item(state) - j.item(state))
            if diff > d:
                d = diff
        if d <= (1-mdp.discount())/mdp.discount(): break
    return j

j = value_iteration(MDP())
print j