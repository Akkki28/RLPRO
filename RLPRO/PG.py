
import torch
from visualize import plot_total_rewards

def PG(policy, optim, env, n_episodes=100, naive=False, gamma=0.99):
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
        if naive:
            for reward in rewards:
                returns.appen(sum(rewards))
        
        else:
            for t in range(len(rewards)):
                G = 0.0
                for k, r in enumerate(rewards[t:]):
                    G += (gamma ** k) * r
                    returns.append(G)
            
        loss = 0
        for log_prob, reward in zip(log_probs, returns):
            loss += -log_prob * reward
        loss = loss / len(rewards)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_rewards_per_episode.append(sum(rewards))

    plot_total_rewards(total_rewards_per_episode)
        
        
         