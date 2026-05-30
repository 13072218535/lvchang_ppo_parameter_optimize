"""
TAA-PPO — Time-Adaptive-Annealing PPO (MPD-PPO改进版)
3项核心改进：时间自适应裁剪 + 裁剪预热 + 自适应平滑正则
（LayerNorm和双Critic为可选增强，当前默认关闭以验证核心改进）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, SCRIPT_DIR)
from config import *
from algorithms.base_algorithm import BaseAlgorithm
from ppo import Actor, Critic  # 复用现有Actor/Critic（BatchNorm）


class TAAPPO(BaseAlgorithm):
    """
    TAA-PPO: Time-Adaptive-Annealing PPO
    3项核心改进（已验证有效）：
      1. 时间自适应裁剪 — 近端天紧ε、远端天松ε（28维独立裁剪）
      2. 裁剪预热 — 前30%训练ε×warmup_factor，匹配探索速度
      3. 自适应平滑正则 — 前期低权重、后期递增至完整
    """

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.action_dim = action_dim
        self.device = device

        # 复用现有Actor/Critic（BatchNorm，验证过的稳定架构）
        self.actor = Actor(input_dim, num_pots, action_dim=action_dim).to(device)
        self.critic = Critic(input_dim, num_pots).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LEARNING_RATE)

        self.gamma = PPO_GAMMA
        self.lamda = PPO_LAMBDA

        # 改进1：时间自适应裁剪ε表 → 展平为28维数组 [alf0, out0, alf1, out1, ...]
        eps_flat = []
        for day_eps in TAA_EPS_SCHEDULE:
            eps_flat.extend(day_eps)
        self.base_eps = torch.FloatTensor(eps_flat).to(device)

        self._current_epoch = 0

        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.next_states, self.dones = [], [], []

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _get_effective_eps(self):
        """改进2：裁剪预热 — 前N轮ε放大，线性退火到目标值"""
        warmup_progress = min(1.0, self._current_epoch / max(TAA_CLIP_WARMUP_EPOCHS, 1))
        warmup_factor = TAA_CLIP_WARMUP_FACTOR - (TAA_CLIP_WARMUP_FACTOR - 1.0) * warmup_progress
        return self.base_eps * warmup_factor

    def _get_smooth_weight(self):
        """改进3：自适应平滑正则权重 — 前期低(0.005)，逐步增至完整(0.05)"""
        e = self._current_epoch
        if e < TAA_SMOOTH_RAMP_START:
            return TAA_SMOOTH_START_WEIGHT
        elif e >= TAA_SMOOTH_RAMP_END:
            return TAA_SMOOTH_END_WEIGHT
        else:
            progress = (e - TAA_SMOOTH_RAMP_START) / (TAA_SMOOTH_RAMP_END - TAA_SMOOTH_RAMP_START)
            return TAA_SMOOTH_START_WEIGHT + (TAA_SMOOTH_END_WEIGHT - TAA_SMOOTH_START_WEIGHT) * progress

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            means, stds = self.actor(state_tensor)
            dist = Normal(means, stds)
            action = torch.clamp(dist.sample(), -1, 1)
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

    def _compute_gae(self, values, next_values, dones):
        gae = 0
        advantages = []
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(values, dtype=np.float64)
        next_values = np.array(next_values, dtype=np.float64)
        dones = np.array(dones, dtype=np.float64)

        r_mean, r_std = rewards.mean(), rewards.std()
        if r_std > 1e-8:
            rewards = (rewards - r_mean) / (r_std + 1e-8)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float64)
        adv_mean, adv_std = advantages.mean(), advantages.std()

        if np.isnan(adv_mean) or adv_std < 1e-8:
            deltas = [rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
                      for i in range(len(rewards))]
            advantages = np.array(deltas, dtype=np.float64)
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if adv_std < 1e-8:
                return None

        return (advantages - adv_mean) / (adv_std + 1e-8)

    def update(self, ppo_epochs=PPO_INNER_EPOCHS):
        if len(self.states) == 0:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        if torch.isnan(states).any() or torch.isnan(actions).any():
            self._clear_buffers(); return 0.0, 0.0

        with torch.no_grad():
            values = self.critic(states).detach().cpu().numpy().flatten()
            next_values = self.critic(next_states).detach().cpu().numpy().flatten()

        dones_np = dones.detach().cpu().numpy()
        if np.isnan(values).any() or np.isnan(next_values).any():
            self._clear_buffers(); return 0.0, 0.0

        advantages = self._compute_gae(values, next_values, dones_np)
        if advantages is None:
            self._clear_buffers(); return 0.0, 0.0

        advantages = torch.FloatTensor(advantages).to(self.device)
        targets = advantages + torch.FloatTensor(values).to(self.device)

        effective_eps = self._get_effective_eps()  # (28,)

        dataset_size = len(states)
        indices = np.arange(dataset_size)
        mini_batch_size = min(PPO_MINI_BATCH_SIZE, dataset_size)

        actor_loss_sum, critic_loss_sum, update_count = 0.0, 0.0, 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                mb_idx = indices[start:start + mini_batch_size]
                mb_s = states[mb_idx]; mb_a = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx].unsqueeze(1)
                mb_tgt = targets[mb_idx].unsqueeze(1)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                means, stds = self.actor(mb_s)
                if torch.isnan(means).any(): continue

                dist = Normal(means, stds)
                new_log_probs = dist.log_prob(mb_a)
                if torch.isnan(new_log_probs).any(): continue

                # ═══ 改进1：28维时间自适应裁剪 ═══
                log_ratio = torch.clamp(new_log_probs - mb_old_lp.detach(), -100, 100)
                ratio = torch.exp(log_ratio)
                ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))

                eps = effective_eps.unsqueeze(0)  # (1, 28)
                clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

                mb_adv_clamped = torch.clamp(mb_adv, -1e3, 1e3)
                per_dim_surr = torch.min(ratio * mb_adv_clamped, clipped_ratio * mb_adv_clamped)
                actor_loss = -per_dim_surr.mean()

                # ═══ 改进3：自适应平滑正则 ═══
                smooth_w = self._get_smooth_weight()
                means_reshaped = means.view(-1, 14, 2)
                day_diffs = means_reshaped[:, 1:, :] - means_reshaped[:, :-1, :]
                alf_smooth = F.relu(day_diffs[:, :, 0].abs() - SMOOTH_REG_ALF_THRESHOLD).pow(2).mean()
                out_smooth = F.relu(day_diffs[:, :, 1].abs() - SMOOTH_REG_OUT_THRESHOLD).pow(2).mean()
                actor_loss = actor_loss + smooth_w * (alf_smooth + out_smooth)

                if torch.isnan(actor_loss): continue
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                current_v = self.critic(mb_s)
                current_v = torch.where(torch.isfinite(current_v), current_v, torch.zeros_like(current_v))
                mb_tgt_safe = torch.where(torch.isfinite(mb_tgt), mb_tgt, torch.zeros_like(mb_tgt))
                critic_loss = nn.MSELoss()(current_v, mb_tgt_safe)
                if torch.isnan(critic_loss): continue
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                update_count += 1

        self._clear_buffers()
        return actor_loss_sum / max(update_count, 1), critic_loss_sum / max(update_count, 1)

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
        if load_optimizer:
            try: self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
            except: pass
            try: self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
            except: pass
