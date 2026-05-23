import gymnasium as gym
import torch
import torch.optiom as optim
from Networks import PolicyNetwork

env = gym.make("CartPole",render_mode="rgb_array")
input_len = env.observation_space.shape[0]
output_len = env.action_space.n

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
num_episodes = 100
    
