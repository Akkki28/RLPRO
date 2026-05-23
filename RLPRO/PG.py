import torch

def PG(policy,optim,n_episodes,env):
    for i in range(n_episodes):
        s = env.reset[0]
        terminated = False
        rewards = []
        log_probs = []
        actions = []
        ep_len = 0
        
        while terminated is False:
            s_tensor = torch.from_numpy(s)
            action_dist = policy.forward(s_tensor)
            a = torch.distibutions.Categorical(action_dist).sample()
            log_prob = torch.distributions.Categorical(action_dist).log_prob(a) #TODO: log_prob
            s,r,terminated,truncated,_ =  env.step(a.item()) #TODO: item????
            rewards.append(r)
            log_probs.append(log_prob)
            actions.appen(a)
            
        returns = []
        sum = 0
        
        for reward in reversed(rewards): ## TODO: tf is goin on here
            sum = sum + reward
            returns.append(sum)
        
        
        # TODO: Exact dynamics of this stuff
        loss = 0
        for log_prob,reward in zip(log_probs,returns):
            loss += -log_prob*rewards
        
        loss = loss/len(rewards)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        
         