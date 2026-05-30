"""
SAC — Soft Actor-Critic (改进版)
4维随机策略 + 轨迹插值扩展 + MLP Q-Critic + 自动α调节
"""
import os, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, SCRIPT_DIR)
from config import *
from algorithms.base_algorithm import BaseAlgorithm, ReplayBuffer, QNetworkMLP
from algorithms.trajectory_utils import TrajectoryExpander


class StochasticReducedActor4D(nn.Module):
    """
    4维随机策略Actor（SAC专用）
    输出 (means_4d, stds_4d) → 采样 → TrajectoryExpander → 28维轨迹
    """

    def __init__(self, input_dim, num_pots, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 pot_embed_dim=POT_EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim

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
        self.mean_head = nn.Linear(hidden_dim // 2, 4)
        self.std_head = nn.Linear(hidden_dim // 2, 4)
        self.softplus = nn.Softplus()
        self._init_weights()

        self.expander = TrajectoryExpander(output_len=14)

    def _init_weights(self):
        for m in self.shared_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in [self.mean_head, self.std_head]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

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
        means_4d = torch.tanh(self.mean_head(shared_out))
        stds_4d = self.softplus(self.std_head(shared_out)) + 1e-4
        return means_4d, stds_4d

    def sample_and_expand(self, state):
        """采样4维动作 → 扩展为28维轨迹"""
        means_4d, stds_4d = self.forward(state)
        dist = Normal(means_4d, stds_4d)
        action_4d = dist.rsample()
        action_4d = torch.clamp(action_4d, -1, 1)
        log_prob_4d = dist.log_prob(action_4d).sum(dim=-1, keepdim=True)  # (batch, 1)
        trajectory_28d = self.expander(action_4d)
        return trajectory_28d, log_prob_4d

    def expand_actions(self, actions_4d):
        return self.expander(actions_4d)


class SAC(BaseAlgorithm):
    """SAC — 4D随机策略 + 双Q + 自动α"""

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
        self.input_dim = input_dim; self.num_pots = num_pots
        self.action_dim = action_dim; self.device = device
        self.state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 4

        self.actor = StochasticReducedActor4D(input_dim, num_pots).to(device)
        self.critic1 = QNetworkMLP(self.state_dim, action_dim).to(device)
        self.critic2 = QNetworkMLP(self.state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1); self.critic1_target.eval()
        self.critic2_target = copy.deepcopy(self.critic2); self.critic2_target.eval()

        self.target_entropy = -4  # 4维动作的目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=PPO_LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=PPO_LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.gamma = PPO_GAMMA; self.tau = 0.005; self._train_step = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            means_4d, stds_4d = self.actor(state_tensor)
            dist = Normal(means_4d, stds_4d)
            action_4d = torch.clamp(dist.sample(), -1, 1)
            trajectory_28d = self.actor.expander(action_4d)
        self.actor.train()
        return trajectory_28d.detach().cpu().numpy().squeeze(0)

    def store_transition(self, state, action, reward, log_prob_or_none, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def update(self):
        if len(self.replay_buffer) < 256: return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(256)
        s = torch.FloatTensor(states).to(self.device); a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic
        with torch.no_grad():
            na, nlp = self.actor.sample_and_expand(ns)
            tq = r + self.gamma * (torch.min(self.critic1_target(ns, na), self.critic2_target(ns, na)) - self.alpha * nlp) * (1 - d)
        cl = nn.MSELoss()(self.critic1(s, a), tq) + nn.MSELoss()(self.critic2(s, a), tq)
        if torch.isnan(cl): return 0.0, 0.0
        self.critic_optimizer.zero_grad(); cl.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor
        self.actor_optimizer.zero_grad()
        aa, alp = self.actor.sample_and_expand(s)
        al = (self.alpha * alp - torch.min(self.critic1(s, aa), self.critic2(s, aa))).mean()
        if torch.isnan(al): return 0.0, cl.item()
        al_val = al.item(); al.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (alp.detach() + self.target_entropy)).mean()
        alpha_loss.backward(); self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        self._train_step += 1
        return al_val, cl.item()

    def save_model(self, path):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                     'critic1_state_dict': self.critic1.state_dict(),
                     'critic2_state_dict': self.critic2.state_dict(),
                     'log_alpha': self.log_alpha.item()}, path)

    def load_model(self, path, load_optimizer=False):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
        self.critic1.load_state_dict(ckpt['critic1_state_dict'], strict=False)
        self.critic2.load_state_dict(ckpt['critic2_state_dict'], strict=False)
        self.critic1_target.load_state_dict(ckpt['critic1_state_dict'], strict=False)
        self.critic2_target.load_state_dict(ckpt['critic2_state_dict'], strict=False)
        if 'log_alpha' in ckpt:
            self.log_alpha.data.fill_(ckpt['log_alpha'])
            self.alpha = self.log_alpha.exp().item()
