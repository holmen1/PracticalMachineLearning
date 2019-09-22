# -*- coding: utf-8 -*-
import numpy as np
import gym

env = gym.make('CartPole-v0').unwrapped

def run_episode(env, parameters):
    print('parameters')
    print(parameters)
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        env.render()
        observation, reward, done, info = env.step(action)
        print('observation')
        print(observation)
        totalreward += reward
        if done:
            break
    return totalreward

parameters = np.random.rand(4) * 2 - 1
bestparams = None  
bestreward = 0  
for _ in range(10000):  
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    print(reward)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            print('Solved!')
            break
        
noise_scaling = 1
parameters = np.random.rand(4) * 2 - 1  
bestreward = 0  
for _ in range(10000):  
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = run_episode(env,newparams)
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 200:
            print('Solved!!')
            break