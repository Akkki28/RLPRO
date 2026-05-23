
import torch
from visualize import plot_total_rewards

def PG(policy, optim, n_episodes, env):
    print("Algorithm: POLICY GRADIENTS")
    total_rewards_per_episode = []
    for i in range(n_episodes):
        s = env.reset()[0]
        terminated = False
        rewards = []
        log_probs = []
        actions = []
        
        while terminated is False:
            s_tensor = torch.from_numpy(s)
            action_dist = policy.forward(s_tensor)
            a = torch.distributions.Categorical(action_dist).sample()
            log_prob = torch.distributions.Categorical(action_dist).log_prob(a)
            s, r, terminated, truncated, info = env.step(a.item())
            rewards.append(r)
            log_probs.append(log_prob)
            actions.append(a)
        
        returns = []
        sumi = 0
        for reward in reversed(rewards):
            sumi = sumi + reward
            returns.append(sumi)
        
        loss = 0
        for log_prob, reward in zip(log_probs, returns):
            loss += -log_prob * reward
        loss = loss / len(rewards)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_rewards_per_episode.append(sum(rewards))

    plot_total_rewards(total_rewards_per_episode)
        
        
         