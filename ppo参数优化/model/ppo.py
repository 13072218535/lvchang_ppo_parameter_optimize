"""
MPD-PPO算法实现
多分支Actor、Critic、差异化裁剪
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from config import *


class Actor(nn.Module):
    """
    多分支策略Actor网络
    包含LSTM编码器处理7天历史窗口，然后分两个独立分支输出动作
    """

    def __init__(self, input_dim, num_pots, pot_embed_dim=POT_EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, action_dim=2,
                 dropout=DROPOUT):
        super().__init__()

        self.input_dim = input_dim
        self.num_pots = num_pots
        self.pot_embed_dim = pot_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim

        self.pot_embedding = nn.Embedding(num_pots, pot_embed_dim)
        nn.init.uniform_(self.pot_embedding.weight, -0.1, 0.1)

        self.lstm = nn.LSTM(
            input_size=input_dim + pot_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.bn = nn.BatchNorm1d(hidden_dim)

        combined_dim = hidden_dim + OUTPUT_LEN + pot_embed_dim
        self.shared_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )

        self.alf_mean_head = nn.Linear(hidden_dim // 2, 1)
        self.alf_std_head = nn.Linear(hidden_dim // 2, 1)

        self.out_mean_head = nn.Linear(hidden_dim // 2, 1)
        self.out_std_head = nn.Linear(hidden_dim // 2, 1)

        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.shared_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in [self.alf_mean_head, self.alf_std_head, self.out_mean_head, self.out_std_head]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        参数:
            state: (batch_size, 7*input_dim + 14 + 1) 展平的状态向量
        返回:
            alf_mean, alf_std, out_mean, out_std
        """
        batch_size = state.shape[0]

        past_features = state[:, :INPUT_LEN * self.input_dim].view(batch_size, INPUT_LEN, self.input_dim)
        target_voltage = state[:, INPUT_LEN * self.input_dim:INPUT_LEN * self.input_dim + OUTPUT_LEN]
        pot_ids = state[:, -1].long()

        if torch.isnan(state).any():
            print("NaN detected in input state!")
            print(f"state has NaN: {torch.isnan(state).sum()} NaN values")
        if torch.isinf(state).any():
            print("Inf detected in input state!")

        pot_embed = self.pot_embedding(pot_ids)
        pot_embed = pot_embed.unsqueeze(1).expand(-1, INPUT_LEN, -1)

        x = torch.cat([past_features, pot_embed], dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]

        if self.training and lstm_last.shape[0] > 1:
            lstm_last = self.bn(lstm_last)

        pot_embed_last = pot_embed[:, -1, :]
        combined = torch.cat([lstm_last, target_voltage, pot_embed_last], dim=1)

        shared_out = self.shared_net(combined)

        alf_mean = torch.tanh(self.alf_mean_head(shared_out))
        alf_std = self.softplus(self.alf_std_head(shared_out)) + 1e-3

        out_mean = torch.tanh(self.out_mean_head(shared_out))
        out_std = self.softplus(self.out_std_head(shared_out)) + 1e-3

        alf_mean = torch.clamp(alf_mean, -10, 10)
        alf_std = torch.clamp(alf_std, 1e-3, 100)
        out_mean = torch.clamp(out_mean, -10, 10)
        out_std = torch.clamp(out_std, 1e-3, 100)

        if torch.isnan(alf_mean).any() or torch.isnan(alf_std).any():
            print(f"NaN in alf_mean: {torch.isnan(alf_mean).sum()}, alf_std: {torch.isnan(alf_std).sum()}")
            print(f"shared_out has NaN: {torch.isnan(shared_out).sum()}")
            print(f"lstm_last has NaN: {torch.isnan(lstm_last).sum()}")

        return alf_mean, alf_std, out_mean, out_std

    def sample_action(self, state):
        """
        采样动作
        参数:
            state: (batch_size, state_dim) 或 (state_dim,)
        返回:
            action: (batch_size, 2) 或 (2,) 归一化到[-1,1]
            log_prob: (batch_size, 2) 两个动作分量的对数概率
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            alf_mean, alf_std, out_mean, out_std = self.forward(state)

            alf_dist = Normal(alf_mean, alf_std)
            out_dist = Normal(out_mean, out_std)

            alf_action = alf_dist.sample()
            out_action = out_dist.sample()

            alf_action = torch.clamp(alf_action, -1, 1)
            out_action = torch.clamp(out_action, -1, 1)

            alf_log_prob = alf_dist.log_prob(alf_action)
            out_log_prob = out_dist.log_prob(out_action)

            log_prob = torch.cat([alf_log_prob, out_log_prob], dim=1)

            action = torch.cat([alf_action, out_action], dim=1)

        self.train()

        return action, log_prob

    def get_log_prob(self, state, action):
        """
        计算给定状态和动作的对数概率
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        alf_mean, alf_std, out_mean, out_std = self.forward(state)

        alf_dist = Normal(alf_mean, alf_std)
        out_dist = Normal(out_mean, out_std)

        alf_log_prob = alf_dist.log_prob(action[:, 0:1])
        out_log_prob = out_dist.log_prob(action[:, 1:2])

        log_prob = torch.cat([alf_log_prob, out_log_prob], dim=1)

        return log_prob.squeeze(1)


class Critic(nn.Module):
    """
    Critic网络 - 输出状态价值V(s)
    同样使用LSTM处理历史窗口
    """

    def __init__(self, input_dim, num_pots, pot_embed_dim=POT_EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.input_dim = input_dim
        self.num_pots = num_pots
        self.pot_embed_dim = pot_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.pot_embedding = nn.Embedding(num_pots, pot_embed_dim)
        nn.init.uniform_(self.pot_embedding.weight, -0.1, 0.1)

        self.lstm = nn.LSTM(
            input_size=input_dim + pot_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.bn = nn.BatchNorm1d(hidden_dim)

        combined_dim = hidden_dim + OUTPUT_LEN + pot_embed_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        参数:
            state: (batch_size, 7*input_dim + 14 + 1) 展平的状态向量
        返回:
            value: (batch_size, 1)
        """
        batch_size = state.shape[0]

        past_features = state[:, :INPUT_LEN * self.input_dim].view(batch_size, INPUT_LEN, self.input_dim)
        target_voltage = state[:, INPUT_LEN * self.input_dim:INPUT_LEN * self.input_dim + OUTPUT_LEN]
        pot_ids = state[:, -1].long()

        pot_embed = self.pot_embedding(pot_ids)
        pot_embed = pot_embed.unsqueeze(1).expand(-1, INPUT_LEN, -1)

        x = torch.cat([past_features, pot_embed], dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]

        if self.training and lstm_last.shape[0] > 1:
            lstm_last = self.bn(lstm_last)

        pot_embed_last = pot_embed[:, -1, :]
        combined = torch.cat([lstm_last, target_voltage, pot_embed_last], dim=1)

        value = self.net(combined)
        value = torch.clamp(value, -1000, 1000)

        return value


class MPDPPO:
    """
    MPD-PPO算法类
    支持差异化裁剪阈值
    """

    def __init__(self, input_dim, num_pots, action_dim=2, device='cpu'):
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.action_dim = action_dim
        self.device = device

        self.actor = Actor(
            input_dim=input_dim,
            num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            action_dim=action_dim,
            dropout=DROPOUT
        ).to(device)

        self.critic = Critic(
            input_dim=input_dim,
            num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_LEARNING_RATE)

        self.eps_clip_alf = EPS_CLIP_ALF
        self.eps_clip_out = EPS_CLIP_OUT

        self.gamma = PPO_GAMMA
        self.lamda = PPO_LAMBDA

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        """
        选择动作
        参数:
            state: (state_dim,) numpy数组
        返回:
            action: (2,) numpy数组，归一化到[-1,1]
            log_prob: (2,) numpy数组，两个动作分量的对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.actor.sample_action(state_tensor)

        action_np = action.detach().cpu().numpy().squeeze(0)
        log_prob_np = log_prob.detach().cpu().numpy().squeeze(0)

        return action_np, log_prob_np
    
    def store_transition(self, state, action, reward, log_prob, next_state, done):
        """
        存储经验
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def compute_gae(self, values, next_values, dones):
        """
        计算广义优势估计（GAE）
        """
        gae = 0
        advantages = []

        values = np.array(values, dtype=np.float64)
        next_values = np.array(next_values, dtype=np.float64)
        dones = np.array(dones, dtype=np.float64)
        rewards = np.array(self.rewards, dtype=np.float64)

        rewards = np.clip(rewards, -1e3, 1e3)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float64)
        advantages = np.clip(advantages, -1e3, 1e3)

        advantages_mean = advantages.mean()
        advantages_std = advantages.std()

        if np.isnan(advantages_mean) or np.isnan(advantages_std) or advantages_std < 1e-8:
            print(f"Invalid advantages stats: mean={advantages_mean}, std={advantages_std}")
            return np.zeros_like(advantages)

        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

        return advantages
    
    def update(self, ppo_epochs=10):
        """
        更新PPO网络
        """
        if len(self.states) == 0:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        if torch.isnan(states).any() or torch.isnan(actions).any():
            print("NaN detected in states or actions, skipping update")
            self._clear_buffers()
            return 0.0, 0.0

        values = self.critic(states).detach().cpu().numpy().flatten()
        next_values = self.critic(next_states).detach().cpu().numpy().flatten()
        dones_np = dones.detach().cpu().numpy()

        if np.isnan(values).any() or np.isnan(next_values).any():
            print("NaN detected in values, skipping update")
            self._clear_buffers()
            return 0.0, 0.0

        advantages = self.compute_gae(values, next_values, dones_np)

        if np.isnan(advantages).any():
            print("NaN detected in advantages, skipping update")
            self._clear_buffers()
            return 0.0, 0.0

        advantages = torch.FloatTensor(advantages).to(self.device)
        targets = advantages + torch.FloatTensor(values).to(self.device)

        if torch.isnan(targets).any():
            print("NaN detected in targets, skipping update")
            self._clear_buffers()
            return 0.0, 0.0

        actor_loss_sum = 0
        critic_loss_sum = 0

        for _ in range(ppo_epochs):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            alf_mean, alf_std, out_mean, out_std = self.actor(states)

            if torch.isnan(alf_mean).any() or torch.isnan(alf_std).any():
                print(f"NaN in actor output at iteration {_}")
                print(f"  alf_mean has nan: {torch.isnan(alf_mean).sum()}, inf: {torch.isinf(alf_mean).sum()}")
                print(f"  alf_std has nan: {torch.isnan(alf_std).sum()}, inf: {torch.isinf(alf_std).sum()}")
                print(f"  out_mean has nan: {torch.isnan(out_mean).sum()}, inf: {torch.isinf(out_mean).sum()}")
                print(f"  out_std has nan: {torch.isnan(out_std).sum()}, inf: {torch.isinf(out_std).sum()}")
                print(f"  states has nan: {torch.isnan(states).sum()}, inf: {torch.isinf(states).sum()}")
                self._clear_buffers()
                return 0.0, 0.0

            alf_dist = Normal(alf_mean, alf_std)
            out_dist = Normal(out_mean, out_std)

            new_alf_log_prob = alf_dist.log_prob(actions[:, 0:1])
            new_out_log_prob = out_dist.log_prob(actions[:, 1:2])

            if torch.isnan(new_alf_log_prob).any() or torch.isnan(new_out_log_prob).any():
                print(f"NaN in log_prob!")
                print(f"  new_alf_log_prob has nan: {torch.isnan(new_alf_log_prob).sum()}")
                print(f"  new_out_log_prob has nan: {torch.isnan(new_out_log_prob).sum()}")
                self._clear_buffers()
                return 0.0, 0.0

            new_log_probs = new_alf_log_prob + new_out_log_prob
            old_log_probs_sum = old_log_probs.sum(dim=1, keepdim=True)

            log_ratio = new_log_probs - old_log_probs_sum.detach()
            log_ratio = torch.clamp(log_ratio, -100, 100)
            ratio = torch.exp(log_ratio)

            ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))

            alf_ratio = torch.exp(torch.clamp(new_alf_log_prob - old_log_probs[:, :1].detach(), -100, 100))
            alf_ratio = torch.where(torch.isfinite(alf_ratio), alf_ratio, torch.ones_like(alf_ratio))
            clipped_alf_ratio = torch.clamp(alf_ratio, 1 - self.eps_clip_alf, 1 + self.eps_clip_alf)

            out_ratio = torch.exp(torch.clamp(new_out_log_prob - old_log_probs[:, 1:].detach(), -100, 100))
            out_ratio = torch.where(torch.isfinite(out_ratio), out_ratio, torch.ones_like(out_ratio))
            clipped_out_ratio = torch.clamp(out_ratio, 1 - self.eps_clip_out, 1 + self.eps_clip_out)

            clipped_ratio = clipped_alf_ratio * clipped_out_ratio

            ratio_safe = torch.clamp(ratio, 0.0, 1e3)
            advantages_tensor = advantages.unsqueeze(1)

            advantages_safe = torch.clamp(advantages_tensor, -1e3, 1e3)

            surr1 = ratio_safe * advantages_safe
            surr2 = clipped_ratio * advantages_safe

            surr1_safe = torch.clamp(surr1, -1e6, 1e6)
            surr2_safe = torch.clamp(surr2, -1e6, 1e6)

            min_surr = torch.min(surr1_safe, surr2_safe)
            neg_min_surr = -min_surr

            if min_surr.numel() == 0:
                print(f"ERROR: min_surr is empty! surr1 shape: {surr1.shape}, surr2 shape: {surr2.shape}")
                actor_loss_sum += 0.0
                continue

            actor_loss = neg_min_surr.mean()

            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                print(f"actor_loss={actor_loss}")
                actor_loss_sum += 0.0
                continue

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            current_values = self.critic(states)
            current_values = torch.where(torch.isfinite(current_values), current_values, torch.zeros_like(current_values))
            targets_safe = torch.where(torch.isfinite(targets.unsqueeze(1)), targets.unsqueeze(1), torch.zeros_like(targets.unsqueeze(1)))
            critic_loss = nn.MSELoss()(current_values, targets_safe)

            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                print(f"Warning: critic_loss is {critic_loss}, using 0 loss")
                critic_loss_sum += 0.0
                continue

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            actor_loss_sum += actor_loss.item()
            critic_loss_sum += critic_loss.item()

        self._clear_buffers()

        avg_actor_loss = actor_loss_sum / ppo_epochs if ppo_epochs > 0 else 0.0
        avg_critic_loss = critic_loss_sum / ppo_epochs if ppo_epochs > 0 else 0.0

        return avg_actor_loss, avg_critic_loss

    def _clear_buffers(self):
        """清空经验缓存"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []
        self.dones = []
    
    def save_model(self, path):
        """
        保存模型
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])