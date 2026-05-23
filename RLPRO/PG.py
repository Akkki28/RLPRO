
import torch
from visualize import plot_total_rewards

def PG(policy, optim, n_episodes, env, naive=False, gamma=0.99, avg_b=False, opt_b=False):
    print("Algorithm: POLICY GRADIENTS")
    total_rewards_per_episode = []
    for i in range(n_episodes):
        s = env.reset()[0]
        terminated = False
        rewards = []
        log_probs = []
        actions = []
        avg_r = 0
        cnt = 0
        
        while terminated is False:
            s_tensor = torch.from_numpy(s)
            action_dist = policy.forward(s_tensor)
            a = torch.distributions.Categorical(action_dist).sample()
            log_prob = torch.distributions.Categorical(action_dist).log_prob(a)
            s, r, terminated, truncated, info = env.step(a.item())
            avg_r = avg_r + (r - avg_r)/(cnt + 1) 
            cnt+=1
            rewards.append(r)
            log_probs.append(log_prob)
            actions.append(a)
        
        returns = []
        if naive:
            returns = [sum(rewards)] * len(rewards)
        
        else:
            for t in range(len(rewards)):
                G = 0.0
                for k, r in enumerate(rewards[t:]):
                    G += (gamma ** k) * r
                returns.append(G)

        if avg_b:
            for j,ret in enumerate(returns):
                returns[j] = ret - avg_r
            
        loss = 0
        for log_prob, reward in zip(log_probs, returns):
            loss = loss - (log_prob * reward)
        loss = loss / len(rewards)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_rewards_per_episode.append(sum(rewards))

    plot_total_rewards(total_rewards_per_episode)
        
        
         