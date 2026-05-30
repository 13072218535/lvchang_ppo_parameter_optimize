"""
TD3 — Twin Delayed DDPG (改进版)
4维动作 + 轨迹插值 + 时序相关噪声 + MLP Q-Critic
"""
import os, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, SCRIPT_DIR)
from config import *
from algorithms.base_algorithm import BaseAlgorithm, ReplayBuffer, QNetworkMLP
from algorithms.trajectory_utils import generate_trajectory_noise
from algorithms.ddpg import ReducedActor4D


class TD3(BaseAlgorithm):
    """TD3 — 4D动作 + 双Q + 延迟更新 + 时序噪声"""

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim; self.num_pots = num_pots
        self.action_dim = action_dim; self.device = device
        self.state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 4

        self.actor = ReducedActor4D(input_dim, num_pots).to(device)
        self.actor_target = copy.deepcopy(self.actor); self.actor_target.eval()

        self.critic1 = QNetworkMLP(self.state_dim, action_dim).to(device)
        self.critic2 = QNetworkMLP(self.state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1); self.critic1_target.eval()
        self.critic2_target = copy.deepcopy(self.critic2); self.critic2_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=PPO_LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.gamma = PPO_GAMMA; self.tau = 0.005
        self.policy_noise = 0.1; self.noise_clip = 0.3; self.policy_delay = 2
        self.noise_std = 0.15; self.noise_rho = 0.85
        self._train_step = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            trajectory_28d, _ = self.actor(state_tensor)
            noise = torch.FloatTensor(
                generate_trajectory_noise(rho=self.noise_rho, noise_std=self.noise_std)
            ).to(self.device)
            action = torch.clamp(trajectory_28d.squeeze(0) + noise, -1, 1)
        self.actor.train()
        return action.detach().cpu().numpy()

    def store_transition(self, state, action, reward, log_prob_or_none, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def update(self):
        if len(self.replay_buffer) < 256: return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(256)
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, _ = self.actor_target(ns)
            noise = torch.clamp(torch.randn_like(na) * self.policy_noise, -self.noise_clip, self.noise_clip)
            na = torch.clamp(na + noise, -1, 1)
            tq = r + self.gamma * torch.min(self.critic1_target(ns, na), self.critic2_target(ns, na)) * (1 - d)

        cl = nn.MSELoss()(self.critic1(s, a), tq) + nn.MSELoss()(self.critic2(s, a), tq)
        if torch.isnan(cl): return 0.0, 0.0
        self.critic_optimizer.zero_grad(); cl.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        al_val = 0.0
        if self._train_step % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            aa, _ = self.actor(s); al = -self.critic1(s, aa).mean()
            if not torch.isnan(al):
                al.backward(); al_val = al.item()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
        self._train_step += 1
        return al_val, cl.item()

    def save_model(self, path):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                     'critic1_state_dict': self.critic1.state_dict(),
                     'critic2_state_dict': self.critic2.state_dict()}, path)

    def load_model(self, path, load_optimizer=False):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.actor_target.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.critic1.load_state_dict(ckpt['critic1_state_dict'], strict=False)
        self.critic2.load_state_dict(ckpt['critic2_state_dict'], strict=False)
        self.critic1_target.load_state_dict(ckpt['critic1_state_dict'], strict=False)
        self.critic2_target.load_state_dict(ckpt['critic2_state_dict'], strict=False)
