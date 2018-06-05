from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(alpha=0.2,gamma=1.0)
avg_rewards, best_avg_reward = interact(env, agent,num_episodes=500000)