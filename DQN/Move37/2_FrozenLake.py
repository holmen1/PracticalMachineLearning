import gym
import sys
import numpy as np
sys.path.append('C:\\Users\\holme\\OneDrive\\Dokument\\frozenlake')
import deeprl_hw1.lake_envs as lake_env
# https://github.com/aaksham/frozenlake.git

"""    
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

P[s][a] = [(prob, nextstate, reward, is_terminal), ...]

"""
class ValueIteration(object):
    """
    
    """
    
    def __init__(self):
        self.env = gym.make('Deterministic-4x4-FrozenLake-v0')
        


    def v(self, s):
        for action in range(self.env.nA):
            vtemp = self.env.P[s][action]       
            print(vtemp)
        
    def valueiter(self, n):
        v = np.zeros(self.env.nS)
        for k in range(n):
            for s in range(self.env.nS):
                vv = 0
                for action in range(self.env.nA):
                    vtemp = self.env.P[s][action]
                    if vtemp[0][2] + v[vtemp[0][1]] > vv:
                        vv = vtemp[0][2] + v[vtemp[0][1]]
                v[s] = vv
        
        print(v)



vi = ValueIteration()

#v = np.zeros(env.nS)
vi.v(0)

vi.valueiter(4)