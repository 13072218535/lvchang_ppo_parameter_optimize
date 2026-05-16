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
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, action_dim=ACTION_TRAJECTORY_DIM,
                 dropout=DROPOUT):
        super().__init__()

        self.input_dim = input_dim
        self.num_pots = num_pots
        self.pot_embed_dim = pot_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim  # 28维（14天×2动作）

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

        # 输出28维动作轨迹：每个维度独立均值与标准差
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.std_head = nn.Linear(hidden_dim // 2, action_dim)

        self.softplus = nn.Softplus()

        self._init_weights()

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧版本模型（4个头→2个头）的参数名映射和形状不匹配处理"""
        old_keys = ['alf_mean_head', 'alf_std_head', 'out_mean_head', 'out_std_head']
        new_keys = ['mean_head', 'std_head', 'mean_head', 'std_head']
        suffix = ['.weight', '.bias']

        mapped = {}
        for old_name, new_name in zip(old_keys, new_keys):
            for s in suffix:
                old_key = old_name + s
                new_key = new_name + s
                if old_key in state_dict and new_key not in state_dict:
                    mapped[new_key] = state_dict.pop(old_key)

        if mapped:
            state_dict.update(mapped)
            print(f"  已映射 {len(mapped) // 2} 组旧版参数名")

        # 过滤形状不匹配的参数（如旧版输出头维度1→新版维度28）
        current = self.state_dict()
        skip_keys = []
        for key in list(state_dict.keys()):
            if key in current:
                if state_dict[key].shape != current[key].shape:
                    print(f"  跳过形状不匹配: {key} "
                          f"checkpoint={tuple(state_dict[key].shape)} "
                          f"model={tuple(current[key].shape)}")
                    skip_keys.append(key)
        for k in skip_keys:
            del state_dict[k]

        return super().load_state_dict(state_dict, strict=False)

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.shared_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in [self.mean_head, self.std_head]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                if m is self.std_head:
                    nn.init.constant_(m.bias, -1.0)  # 初始std≈0.31，避免过度随机
                else:
                    nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        参数:
            state: (batch_size, 7*input_dim + 14 + 1) 展平的状态向量
        返回:
            means: (batch_size, 28) 28维动作均值
            stds: (batch_size, 28) 28维动作标准差
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

        shared_out = self.shared_net(combined)

        means = torch.tanh(self.mean_head(shared_out))
        stds = self.softplus(self.std_head(shared_out)) + 1e-3

        means = torch.clamp(means, -10, 10)
        stds = torch.clamp(stds, 1e-3, 100)

        return means, stds

    def sample_action(self, state):
        """
        采样动作（环境交互用，无梯度）
        参数:
            state: (batch_size, state_dim) 或 (state_dim,)
        返回:
            action: (batch_size, 28) 或 (28,) 归一化到[-1,1]
            log_prob: (batch_size, 28) 28个维度的对数概率
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            means, stds = self.forward(state)

            dist = Normal(means, stds)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action)

        self.train()

        return action, log_prob

    def get_log_prob(self, state, action):
        """
        计算给定状态和动作的对数概率（更新用，有梯度）
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        means, stds = self.forward(state)

        dist = Normal(means, stds)
        log_prob = dist.log_prob(action)

        return log_prob


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
        value = torch.clamp(value, -50, 50)

        return value


class MPDPPO:
    """
    MPD-PPO算法类
    支持差异化裁剪阈值
    """

    def __init__(self, input_dim, num_pots, action_dim=ACTION_TRAJECTORY_DIM, device='cpu'):
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
            action: (28,) numpy数组，归一化到[-1,1]
            log_prob: (28,) numpy数组，各维度对数概率
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

        # 对rewards做标准化（而非硬裁剪），保留相对好坏信号
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        if rewards_std > 1e-8:
            rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float64)

        advantages_mean = advantages.mean()
        advantages_std = advantages.std()

        if np.isnan(advantages_mean) or np.isnan(advantages_std) or advantages_std < 1e-8:
            print(f"Warning: GAE collapsed (mean={advantages_mean:.2f}, std={advantages_std:.6f}), "
                  f"using TD-errors instead")
            deltas = []
            for i in range(len(rewards)):
                delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
                deltas.append(delta)
            advantages = np.array(deltas, dtype=np.float64)
            advantages_mean = advantages.mean()
            advantages_std = advantages.std()
            if advantages_std < 1e-8:
                print("   TD-errors also collapsed, skipping update")
                return None

        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

        return advantages
    
    def update(self, ppo_epochs=PPO_INNER_EPOCHS):
        """
        更新PPO网络（独立裁剪+求和，mini-batch支持）
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

        if advantages is None:
            print("GAE returned None (collapsed), skipping update")
            self._clear_buffers()
            return 0.0, 0.0

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

        # 构建mini-batch索引
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        mini_batch_size = min(PPO_MINI_BATCH_SIZE, dataset_size)

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        update_count = 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, mini_batch_size):
                mb_indices = indices[start:start + mini_batch_size]
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_targets = targets[mb_indices]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                means, stds = self.actor(mb_states)

                if torch.isnan(means).any() or torch.isnan(stds).any():
                    continue

                dist = Normal(means, stds)
                new_log_probs = dist.log_prob(mb_actions)

                if torch.isnan(new_log_probs).any():
                    continue

                # === 独立裁剪 + 求和 ===
                # 偶数索引(0,2,4,...,26)：ALF维度，使用eps_clip_alf=0.1
                # 奇数索引(1,3,5,...,27)：OUT维度，使用eps_clip_out=0.2
                alf_indices = list(range(0, self.action_dim, 2))
                out_indices = list(range(1, self.action_dim, 2))

                # ALF ratio + clipped
                alf_log_ratio = new_log_probs[:, alf_indices] - mb_old_log_probs[:, alf_indices].detach()
                alf_log_ratio = torch.clamp(alf_log_ratio, -100, 100)
                alf_ratio = torch.exp(alf_log_ratio)
                alf_ratio = torch.where(torch.isfinite(alf_ratio), alf_ratio, torch.ones_like(alf_ratio))
                clipped_alf_ratio = torch.clamp(alf_ratio, 1 - self.eps_clip_alf, 1 + self.eps_clip_alf)

                # OUT ratio + clipped
                out_log_ratio = new_log_probs[:, out_indices] - mb_old_log_probs[:, out_indices].detach()
                out_log_ratio = torch.clamp(out_log_ratio, -100, 100)
                out_ratio = torch.exp(out_log_ratio)
                out_ratio = torch.where(torch.isfinite(out_ratio), out_ratio, torch.ones_like(out_ratio))
                clipped_out_ratio = torch.clamp(out_ratio, 1 - self.eps_clip_out, 1 + self.eps_clip_out)

                mb_adv = mb_advantages.unsqueeze(1)
                mb_adv = torch.clamp(mb_adv, -1e3, 1e3)

                # ALF surrogate loss（独立均值）
                alf_surr1 = alf_ratio * mb_adv
                alf_surr2 = clipped_alf_ratio * mb_adv
                alf_surr = torch.min(alf_surr1, alf_surr2).mean()

                # OUT surrogate loss（独立均值）
                out_surr1 = out_ratio * mb_adv
                out_surr2 = clipped_out_ratio * mb_adv
                out_surr = torch.min(out_surr1, out_surr2).mean()

                # 求和：Actor loss = -(alf_surr + out_surr)
                actor_loss = -(alf_surr + out_surr)

                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    continue

                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                # Critic loss
                current_values = self.critic(mb_states)
                current_values = torch.where(torch.isfinite(current_values), current_values, torch.zeros_like(current_values))
                mb_targets_safe = torch.where(
                    torch.isfinite(mb_targets.unsqueeze(1)),
                    mb_targets.unsqueeze(1),
                    torch.zeros_like(mb_targets.unsqueeze(1))
                )
                critic_loss = nn.MSELoss()(current_values, mb_targets_safe)

                if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                    continue

                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                update_count += 1

        self._clear_buffers()

        avg_actor_loss = actor_loss_sum / max(update_count, 1)
        avg_critic_loss = critic_loss_sum / max(update_count, 1)

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
    
    def load_model(self, path, load_optimizer=False):
        """
        加载模型
        参数:
            path: 模型文件路径
            load_optimizer: 是否加载优化器状态（默认False，评估时不需要）
        """
        checkpoint = torch.load(path, map_location=self.device)

        # 加载Actor权重（含旧版兼容映射）
        missing, unexpected = self.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
        if missing:
            print(f"  Actor missing keys (expected for old model): {missing}")
        if unexpected:
            # 检查是否包含旧版参数名（已在load_state_dict中处理）
            old_heads = ['alf_mean_head', 'alf_std_head', 'out_mean_head', 'out_std_head']
            remaining = [k for k in unexpected if not any(h in k for h in old_heads)]
            if remaining:
                print(f"  Actor unexpected keys: {remaining}")

        # 加载Critic权重
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)

        # 可选加载优化器
        if load_optimizer and 'actor_optimizer_state_dict' in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            except Exception as e:
                print(f"  跳过优化器加载: {e}")
            try:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            except Exception as e:
                print(f"  跳过优化器加载: {e}")