"""
算法基类 + 共享网络组件
定义统一接口 + ReplayBuffer + QNetwork
"""
import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import deque
import random

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, SCRIPT_DIR)
from config import *


class BaseAlgorithm(ABC):
    """所有对比算法的统一接口"""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作（28维，[-1,1]），返回 (action_np,) 或 (action_np, log_prob_np)"""
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, log_prob_or_none, next_state, done):
        """存储经验"""
        pass

    @abstractmethod
    def update(self):
        """更新网络，返回 (actor_loss, critic_loss)"""
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass


class ReplayBuffer:
    """经验回放池（Off-policy算法共用）"""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetworkMLP(nn.Module):
    """
    轻量级Q-Critic网络（Off-policy算法共用）
    输入: state(102) + action(28) → 展平 → MLP → Q-value(1)

    相比LSTM版本的QNetwork，MLP版本10-50x更快，适合off-policy高频更新。
    Q函数无需时序建模，state展平后直接输入MLP即可。
    """

    def __init__(self, state_dim, action_dim=ACTION_TRAJECTORY_DIM, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# 保持向后兼容的别名
QNetwork = QNetworkMLP


def create_actor(input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
    """创建Actor网络（所有算法共用结构）"""
    from ppo import Actor
    actor = Actor(
        input_dim=input_dim,
        num_pots=num_pots,
        pot_embed_dim=POT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        action_dim=action_dim,
        dropout=DROPOUT
    ).to(device)
    actor.lstm.flatten_parameters()  # 抑制RNN weight warning
    return actor


def create_critic(input_dim, num_pots, device='cpu'):
    """创建V-Critic网络（On-policy算法共用）"""
    from ppo import Critic
    return Critic(
        input_dim=input_dim,
        num_pots=num_pots,
        pot_embed_dim=POT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
