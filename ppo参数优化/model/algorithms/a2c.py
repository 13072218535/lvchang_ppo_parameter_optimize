"""
A2C — Advantage Actor-Critic（无裁剪纯策略梯度 + 熵正则）
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, SCRIPT_DIR)
from config import *
from algorithms.base_algorithm import BaseAlgorithm, create_actor, create_critic


class A2C(BaseAlgorithm):
    """A2C — 纯策略梯度，无裁剪"""

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.action_dim = action_dim
        self.device = device
        self.entropy_beta = 0.01  # 熵正则系数

        self.actor = create_actor(input_dim, num_pots, action_dim, device)
        self.critic = create_critic(input_dim, num_pots, device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LEARNING_RATE)

        self.gamma = PPO_GAMMA
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.next_states, self.dones = [], [], []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            means, stds = self.actor(state_tensor)
            dist = Normal(means, stds)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action)
        self.actor.train()
        return action.detach().cpu().numpy().squeeze(0), log_prob.detach().cpu().numpy().squeeze(0)

    def store_transition(self, state, action, reward, log_prob, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.dones.append(done)

    def _compute_returns_and_advantages(self, values, next_values, dones):
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(values, dtype=np.float64)
        next_values = np.array(next_values, dtype=np.float64)
        dones = np.array(dones, dtype=np.float64)

        r_mean, r_std = rewards.mean(), rewards.std()
        if r_std > 1e-8:
            rewards = (rewards - r_mean) / (r_std + 1e-8)

        advantages = []
        for i in range(len(rewards)):
            td_target = rewards[i] + self.gamma * next_values[i] * (1 - dones[i])
            advantages.append(td_target - values[i])

        advantages = np.array(advantages, dtype=np.float64)
        adv_mean, adv_std = advantages.mean(), advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        targets = values + advantages
        return targets, advantages

    def update(self):
        if len(self.states) == 0:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        if torch.isnan(states).any() or torch.isnan(actions).any():
            self._clear_buffers()
            return 0.0, 0.0

        with torch.no_grad():
            values = self.critic(states).detach().cpu().numpy().flatten()
            next_values = self.critic(next_states).detach().cpu().numpy().flatten()
            dones_np = dones.detach().cpu().numpy()

        if np.isnan(values).any() or np.isnan(next_values).any():
            self._clear_buffers()
            return 0.0, 0.0

        targets, advantages = self._compute_returns_and_advantages(values, next_values, dones_np)
        advantages = torch.FloatTensor(advantages).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        means, stds = self.actor(states)
        if torch.isnan(means).any():
            self._clear_buffers()
            return 0.0, 0.0

        dist = Normal(means, stds)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # A2C: 纯策略梯度 + 熵正则
        actor_loss = -(new_log_probs * advantages.unsqueeze(1)).mean()
        actor_loss = actor_loss - self.entropy_beta * entropy

        if torch.isnan(actor_loss):
            self._clear_buffers()
            return 0.0, 0.0

        current_v = self.critic(states)
        critic_loss = nn.MSELoss()(current_v, targets.unsqueeze(1))

        # A2C使用联合优化
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self._clear_buffers()
        return actor_loss.item(), critic_loss.item()

    def _clear_buffers(self):
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.log_probs.clear(); self.next_states.clear(); self.dones.clear()

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path, load_optimizer=False):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.critic.load_state_dict(ckpt['critic_state_dict'], strict=False)
        if load_optimizer and 'actor_optimizer_state_dict' in ckpt:
            try: self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
            except: pass
            try: self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
            except: pass
