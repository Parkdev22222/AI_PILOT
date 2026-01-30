import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical, OneHotCategorical
import os
import glob
import time
from datetime import datetime
import numpy as np
import gym
import statistics
import copy
#from ..utils.fnn import FNN
from clam.utils.fnn import FNN, VQ_FNN


################################## set device ##################################
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")


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
    def __init__(self, state_dim, action_dim, policy_dim, has_vq,has_continuous_action_space, action_std_init,n_rules=10,device=None,order=1):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.has_vq=has_vq
        self.order=order
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            # self.actor = nn.Sequential(
            #     nn.Linear(state_dim, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, action_dim),
            #     nn.Softmax(dim=-1)
            # )
            if self.has_vq:
                self.actor = VQ_FNN(n_rules,policy_dim, state_dim, action_dim, k_clusters=None, stds=None,device=device,order=order)
            else:
                self.actor = nn.Sequential(
                    FNN(policy_dim, state_dim, action_dim, k_clusters=None, stds=None),
                    nn.Softmax(dim=-1)
                )
            # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim+policy_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            if self.has_vq:
                action_mean, vq_loss = self.actor(state)
            else:
                action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            if self.has_vq:
                action_probs,vq_loss = self.actor(state)
            else:
                action_probs = self.actor(state)
            dist = Categorical(action_probs)
            # dist = OneHotCategorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if self.has_vq:
            return action.detach(), action_logprob.detach(), vq_loss
        else:
            return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            if self.has_vq:
                action_mean,vq_loss = self.actor(state)
            else:
                action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            if self.has_vq:
                action_probs,vq_loss = self.actor(state)
            else:
                action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        if self.has_vq:
            return action_logprobs, state_values, dist_entropy,vq_loss
        else:
            return action_logprobs, state_values, dist_entropy


class PPO_FNN:
    def __init__(self, device,state_dim, action_dim, policy_dim, has_vq,lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,n_rules=10,order=1):
        self.device=device
        self.order=order
        self.state_dim=state_dim
        self.action_dim = action_dim
        self.policy_dim=policy_dim
        self.has_continuous_action_space = has_continuous_action_space
        self.has_vq=has_vq
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, policy_dim, has_vq, has_continuous_action_space, action_std_init,n_rules=n_rules,device=device,order=order).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, policy_dim, has_vq, has_continuous_action_space, action_std_init,n_rules=n_rules,device=device,order=order).to(device)
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
                state = torch.FloatTensor(state).to(self.device)
                if self.has_vq:
                    action, action_logprob, vq_loss = self.policy_old.act(state)
                else:
                    action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            if self.has_vq:
                return action.detach().cpu().numpy().flatten(), vq_loss
            else:
                return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                if self.has_vq:
                    action, action_logprob, vq_loss = self.policy_old.act(state)
                else:
                    action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            if self.has_vq:
                return action.item(), vq_loss
            else:
                return action.item()
    def select_action_eval(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                if self.has_vq:
                    action, action_logprob, vq_loss = self.policy_old.act(state)
                else:
                    action, action_logprob = self.policy_old.act(state)

            if self.has_vq:
                return action.detach().cpu().numpy().flatten(), vq_loss
            else:
                return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                if self.has_vq:
                    action, action_logprob, vq_loss = self.policy_old.act(state)
                else:
                    action, action_logprob = self.policy_old.act(state)

            if self.has_vq:
                return action.item(), vq_loss
            else:
                return action.item()
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            # Evaluating old actions and values
            if self.has_vq:
                logprobs, state_values, dist_entropy,vq_loss = self.policy.evaluate(old_states, old_actions)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            if self.has_vq:
                loss+=vq_loss
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        if self.has_vq:
            self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.policy_dim, self.has_vq, self.has_continuous_action_space,
                                          self.action_std, n_rules=len(self.policy.actor.rules), device=self.device, order=self.order).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy_old = copy.deepcopy(self.policy)


        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
