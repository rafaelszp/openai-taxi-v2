import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6,alpha=0.01,gamma=0.999):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.0
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.__action_probs__(self.Q[state])
        #return np.random.choice(self.nA)
        return np.random.choice(range(self.nA),p=probs)
    
    def __action_probs__(self,qs):
        probs = np.full(self.nA,self.epsilon/self.nA)
        best_action_ix = np.argmax(qs)
        e_greedy = (1-self.epsilon)+(self.epsilon/self.nA)
        probs[best_action_ix]=e_greedy
        return probs


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        policy = self.__action_probs__(self.Q[next_state])
        next_vsa = np.dot(self.Q[next_state],policy)
        self.Q[state][action] = self.Q[state][action] * self.alpha*(reward+self.gamma*next_vsa-self.Q[state][action])