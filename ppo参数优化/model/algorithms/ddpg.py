"""
DDPG — Deep Deterministic Policy Gradient
改进版: 4维动作 + 轨迹插值扩展 + 时序相关噪声
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, SCRIPT_DIR)
from config import *
from algorithms.base_algorithm import BaseAlgorithm, ReplayBuffer, QNetworkMLP
from algorithms.trajectory_utils import TrajectoryExpander, generate_trajectory_noise


class ReducedActor4D(nn.Module):
    """
    4维Actor（改进2）
    输出: [alf_start, alf_end, out_start, out_end] → 经TrajectoryExpander变为28维
    复用LSTM编码器处理7天历史窗口，仅输出头维度从28→4
    """

    def __init__(self, input_dim, num_pots, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 pot_embed_dim=POT_EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.pot_embedding = nn.Embedding(num_pots, pot_embed_dim)
        nn.init.uniform_(self.pot_embedding.weight, -0.1, 0.1)

        self.lstm = nn.LSTM(
            input_size=input_dim + pot_embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
        self.lstm.flatten_parameters()

        self.bn = nn.BatchNorm1d(hidden_dim)
        combined_dim = hidden_dim + OUTPUT_LEN + pot_embed_dim
        self.shared_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh()
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 4)  # 仅4维
        self._init_weights()

        self.expander = TrajectoryExpander(output_len=14)

    def _init_weights(self):
        for m in self.shared_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, state):
        batch_size = state.shape[0]
        past_features = state[:, :INPUT_LEN * self.input_dim].view(batch_size, INPUT_LEN, self.input_dim)
        target_voltage = state[:, INPUT_LEN * self.input_dim:INPUT_LEN * self.input_dim + OUTPUT_LEN]
        pot_ids = state[:, -1].long()

        pot_embed = self.pot_embedding(pot_ids).unsqueeze(1).expand(-1, INPUT_LEN, -1)
        x = torch.cat([past_features, pot_embed], dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        if self.training and lstm_last.shape[0] > 1:
            lstm_last = self.bn(lstm_last)

        combined = torch.cat([lstm_last, target_voltage, pot_embed[:, -1, :]], dim=1)
        shared_out = self.shared_net(combined)
        means_4d = torch.tanh(self.mean_head(shared_out))  # (batch, 4)
        trajectory_28d = self.expander(means_4d)  # (batch, 28)
        return trajectory_28d, None  # 兼容DDPG接口（无std）


class DDPG(BaseAlgorithm):
    """DDPG — 4维动作 + 时序相关噪声 + MLP Q-Critic"""

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.action_dim = action_dim
        self.device = device
        self.state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 4  # 102

        self.actor = ReducedActor4D(input_dim, num_pots).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()

        self.critic = QNetworkMLP(self.state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.gamma = PPO_GAMMA
        self.tau = 0.005
        self.noise_std = 0.15
        self.noise_rho = 0.85  # 时序相关系数
        self._train_step = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            trajectory_28d, _ = self.actor(state_tensor)
            # 改进1：时序相关噪声（替代i.i.d.高斯）
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
        if len(self.replay_buffer) < 256:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(256)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + self.gamma * target_q * (1 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_q)
        if torch.isnan(critic_loss):
            return 0.0, 0.0

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_actions, _ = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()
        if torch.isnan(actor_loss):
            return 0.0, critic_loss.item()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        self._train_step += 1
        return actor_loss.item(), critic_loss.item()

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

    def load_model(self, path, load_optimizer=False):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.critic.load_state_dict(ckpt['critic_state_dict'], strict=False)
        self.actor_target.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.critic_target.load_state_dict(ckpt['critic_state_dict'], strict=False)
