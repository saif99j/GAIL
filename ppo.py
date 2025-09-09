import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cuda')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten(), action_logprob

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item(), action_logprob


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
def train_ppo(env, state_dim, action_dim, has_continuous_action_space=False, max_ep_len=1000, 
              max_training_timesteps=3e6, max_training_episode=3e4, save_model_freq=1e4,
              update_timestep=4000, K_epochs=40, eps_clip=0.2, gamma=0.99, lr_actor=0.0003, 
              lr_critic=0.001, action_std=None, render=True, record=True, restore=True,
              checkpoint_path='env4_ppo.ckpt', log_f_path=None, roll_f_path=None):
    
    print("training environment name : " + env.name if hasattr(env, 'name') else "environment")
    
    # initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    
    # restore model if requested
    if restore and checkpoint_path:
        print("restoring model at : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)
        print("model restored")
    
    # logging variables
    time_step = 0
    i_episode = 0
    success_counter = 0
    
    # create logging files
    if log_f_path:
        log_f = open(log_f_path, "w+")
        log_f.write('episode,timestep,reward\n')
    
    if record and roll_f_path:
        rollout = []
        roll_f = open(roll_f_path, "w+")
        roll_f.write('state\taction\tlogprob\treward\tdone\tepisode\tposition\torientation\tspeed\n')
    
    # training loop
    while time_step <= max_training_timesteps:
        
        # reset environment
        state = env.reset()
        step = 0
        current_ep_reward = 0
        
        print("episode: {}".format(i_episode), "success_counter: {}".format(success_counter))
        if i_episode >= max_training_episode:
            print("----------i_episode > max_training_episode---------")
            break

        for t in range(1, max_ep_len+1):

            # select action with policy
            action, logprob = ppo_agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # if render, visualize image
            if render:
                env.render()
            
            # recording information to plot
            if record and roll_f_path:
                roll_f.write('{}\\t{}\\t{}\\t{}\\t{}\\t{}\\t{}\\t{}\\t{}\\n'.format(state, [action], logprob, reward, done, i_episode,
                                                               env.position, env.orientation, env.speed))
                roll_f.flush()
                
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # save model weights
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                print("model saved")
            
            # step of current episodes
            state = state_
            step = t
            
            # break; if the episode is over
            if done:
                break

        # track success
        if current_ep_reward > 120:
            success_counter += 1

        # log in logging file
        if log_f_path:
            log_f.write('{},{},{}\\n'.format(i_episode, step, current_ep_reward))
            log_f.flush()

        i_episode += 1

    if roll_f_path:
        roll_f.close()
    if log_f_path:
        log_f.close()
        
    return ppo_agent

def test_ppo(env, ppo_agent, max_ep_len=1000, render=True):
    """
    Test the trained PPO agent
    """
    state = env.reset()
    current_ep_reward = 0
    
    for t in range(1, max_ep_len+1):
        # select action with policy
        action, _ = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        current_ep_reward += reward
        
        # if render, visualize image
        if render:
            env.render()
            
        if done:
            break
            
    print("Test episode reward:", current_ep_reward)
    return current_ep_reward

# Usage example (commented out):

if __name__ == "__main__":
    from env import Env, envs
    
    env_name = "env4"
    has_continuous_action_space = False
    
    max_ep_len = int(3e3)                    # max timesteps in one episode
    max_training_timesteps = int(5e7)   # break training loop if timeteps > max_training_timesteps
    max_training_episode = int(3e4)
    save_model_freq = int(1e4)      # save model frequency (in num timesteps)
    
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 4               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.98                # discount factor
    
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    
    # initialize environment
    position = envs[env_name]['position']
    orientation = envs[env_name]['orientation']
    speed = envs[env_name]['speed']
    env = Env(position, orientation, speed)
    
    # state space dimension
    state_dim = 4 + 5 * (len(orientation) - 1)
    
    # action space dimension
    action_dim = 3
    
    # train the model
    checkpoint_path = env_name + '_ppo.ckpt'
    log_f_path = 'ppo_' + env_name + '_log.csv'
    roll_f_path = env_name + '_ppo_rollout.csv'
    
    ppo_agent = train_ppo(env, state_dim, action_dim, has_continuous_action_space,
                        max_ep_len, max_training_timesteps, max_training_episode,
                        save_model_freq, update_timestep, K_epochs, eps_clip, gamma,
                        lr_actor, lr_critic, None, True, True, True,
                        checkpoint_path, log_f_path, roll_f_path)
    
    # test the trained model
    test_ppo(env, ppo_agent)
