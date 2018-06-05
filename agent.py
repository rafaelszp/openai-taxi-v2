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
        self.episode=0

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
        return np.random.choice(range(self.nA),p=probs)
        #return np.argmax(self.Q[state]+np.random.randn(1,self.nA)*(1.0/float(self.episode+1)))
    
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
        
        #eps=1.0/(self.episode/20000.0+1.0)
        self.epsilon=0.005
        

        #policy = self.__action_probs__(self.Q[next_state])
        #term = np.dot(self.Q[next_state],policy) # expected sarsa
        term =  0 if done else np.max(self.Q[next_state]) #qlearning
        self.Q[state][action] = self.Q[state][action] + self.alpha*(reward+self.gamma*term-self.Q[state][action])