import gymnasium as gym
import torch
import torch.optim as optim
from Networks import PolicyNetwork
from RLPRO.PG import PG

env = gym.make("CartPole",render_mode="rgb_array")
input_len = env.observation_space.shape[0]
output_len = env.action_space.n

policy = PolicyNetwork(input_len,output_len)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
num_episodes = 100

PG(policy,optimizer,num_episodes,env)
