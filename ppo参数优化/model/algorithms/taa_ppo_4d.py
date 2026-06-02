"""
TAA-PPO-4D V2 — 时间自适应PPO + 4维降维 + LSTM Critic

核心设计：
  - Actor: StochasticReducedActor4D → 4维(means, stds) → TrajectoryExpander → 28维
  - Critic: 复用 ppo.Critic (LSTM + 槽嵌入)，与 MPD-PPO/TAA-PPO 同等可靠
  - 4维 ε per-index 裁剪 + 预热，线性插值自带平滑
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
from algorithms.sac import StochasticReducedActor4D
from ppo import Critic


# ═══════════════════════════════════════════════════════════════
# TAA-PPO-4D V2 (LSTM Critic 版)
# ═══════════════════════════════════════════════════════════════

class TAAPPO4D(BaseAlgorithm):
    """TAA-PPO-4D V2: 4D Actor + LSTM Critic (复用 ppo.Critic)."""

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.action_dim = action_dim
        self.device = device

        self.state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 2 + 1 + 1

        self.actor = StochasticReducedActor4D(input_dim, num_pots).to(device)
        self.critic = Critic(input_dim, num_pots).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LEARNING_RATE)

        self.gamma = PPO_GAMMA
        self.lamda = PPO_LAMBDA

        self.base_eps = torch.FloatTensor(TAA_4D_EPS).to(device)

        self._current_epoch = 0

        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.next_states, self.dones = [], [], []

    # ═══════════ 调度方法 ═══════════

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _get_effective_eps(self):
        warmup_progress = min(1.0, self._current_epoch / max(TAA_CLIP_WARMUP_EPOCHS, 1))
        warmup_factor = TAA_CLIP_WARMUP_FACTOR - (TAA_CLIP_WARMUP_FACTOR - 1.0) * warmup_progress
        return self.base_eps * warmup_factor

    # ═══════════ 交互接口 ═══════════

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            means_4d, stds_4d = self.actor(state_tensor)
            dist = Normal(means_4d, stds_4d)
            action_4d = torch.clamp(dist.sample(), -1, 1)
            log_prob_4d = dist.log_prob(action_4d)
            trajectory_28d = self.actor.expander(action_4d)
        self.actor.train()
        return (trajectory_28d.detach().cpu().numpy().squeeze(0),
                log_prob_4d.detach().cpu().numpy().squeeze(0))

    def store_transition(self, state, action, reward, log_prob, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.dones.append(done)

    # ═══════════ 工具方法 ═══════════

    def _extract_4d_action(self, trajectory_28d):
        return torch.stack([
            trajectory_28d[:, 0],
            trajectory_28d[:, 26],
            trajectory_28d[:, 1],
            trajectory_28d[:, 27],
        ], dim=1)

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

    # ═══════════ PPO 更新（混合 on-policy + replay） ═══════════

    def update(self, ppo_epochs=PPO_INNER_EPOCHS):
        if len(self.states) == 0:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_28d = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        if torch.isnan(states).any() or torch.isnan(actions_28d).any():
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

        actions_4d = self._extract_4d_action(actions_28d)
        effective_eps = self._get_effective_eps()

        dataset_size = len(states)
        indices = np.arange(dataset_size)
        mini_batch_size = min(PPO_MINI_BATCH_SIZE, dataset_size)

        actor_loss_sum, critic_loss_sum, update_count = 0.0, 0.0, 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                mb_idx = indices[start:start + mini_batch_size]
                mb_s = states[mb_idx]
                mb_a_4d = actions_4d[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx].unsqueeze(1)
                mb_tgt = targets[mb_idx].unsqueeze(1)

                # ── Actor: 4维 PPO clip ──
                means_4d, stds_4d = self.actor(mb_s)
                if torch.isnan(means_4d).any(): continue

                dist = Normal(means_4d, stds_4d)
                new_log_probs = dist.log_prob(mb_a_4d)
                if torch.isnan(new_log_probs).any(): continue

                log_ratio = torch.clamp(new_log_probs - mb_old_lp.detach(), -100, 100)
                ratio = torch.exp(log_ratio)
                ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))

                eps = effective_eps.unsqueeze(0)
                clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

                mb_adv_clamped = torch.clamp(mb_adv, -1e3, 1e3)
                per_dim_surr = torch.min(ratio * mb_adv_clamped, clipped_ratio * mb_adv_clamped)
                actor_loss = -per_dim_surr.mean()

                if torch.isnan(actor_loss): continue

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                # ── Critic: on-policy GAE MSE ──
                current_v = self.critic(mb_s)
                current_v = torch.where(torch.isfinite(current_v), current_v, torch.zeros_like(current_v))
                mb_tgt_safe = torch.where(torch.isfinite(mb_tgt), mb_tgt, torch.zeros_like(mb_tgt))
                critic_loss = F.mse_loss(current_v, mb_tgt_safe)
                if torch.isnan(critic_loss): continue

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                update_count += 1

        self._clear_buffers()
        return actor_loss_sum / max(update_count, 1), critic_loss_sum / max(update_count, 1)

    # ═══════════ 缓冲区管理 ═══════════

    def _clear_buffers(self):
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.log_probs.clear(); self.next_states.clear(); self.dones.clear()

    # ═══════════ 持久化 ═══════════

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
