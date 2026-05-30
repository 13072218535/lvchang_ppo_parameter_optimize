"""
Off-policy算法轨迹工具
改进1: 时序相关噪声（替代i.i.d.高斯噪声，保持日间相关性）
改进2: 4维动作→28维轨迹插值扩展器
"""
import numpy as np
import torch
import torch.nn as nn


def generate_correlated_noise(size, rho=0.85, noise_std=0.2):
    """
    生成时序相关的探索噪声（改进1）
    噪声满足 AR(1) 过程: n_t = ρ·n_{t-1} + √(1-ρ²)·ε_t

    参数:
        size: 噪声维度 (例如 14=14天, 或 28=14天×2动作)
        rho: 时序相关系数 (0=无相关, 1=完全平滑)
        noise_std: 噪声标准差
    返回:
        (size,) numpy数组，时序平滑噪声
    """
    noise = np.zeros(size, dtype=np.float32)
    eps = np.random.randn(size).astype(np.float32) * noise_std
    noise[0] = eps[0] * np.sqrt(1 - rho ** 2)  # 初始值方差与后续一致
    for t in range(1, size):
        noise[t] = rho * noise[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]
    return noise


def generate_trajectory_noise(rho=0.85, noise_std=0.15):
    """
    生成28维(14天×2动作)时序相关噪声
    ALF和OUT各14天独立生成，但各自保持时序相关性
    """
    alf_noise = generate_correlated_noise(14, rho, noise_std)
    out_noise = generate_correlated_noise(14, rho, noise_std)
    # 交织: [alf1, out1, alf2, out2, ...]
    trajectory_noise = np.empty(28, dtype=np.float32)
    trajectory_noise[0::2] = alf_noise
    trajectory_noise[1::2] = out_noise
    return trajectory_noise


class TrajectoryExpander(nn.Module):
    """
    动作轨迹扩展器（改进2）
    将4维动作 (alf_start, alf_end, out_start, out_end) 线性插值为28维轨迹

    使用场景: 减少有效动作维度从28→4，使off-policy Q-learning可行
    """

    def __init__(self, output_len=14):
        super().__init__()
        self.output_len = output_len
        # 预计算插值权重: 14天均匀分布
        t = torch.linspace(0, 1, output_len).unsqueeze(1)  # (14, 1)
        self.register_buffer('t', t)

    def forward(self, action_4d):
        """
        参数:
            action_4d: (batch, 4) [alf_start, alf_end, out_start, out_end]
        返回:
            trajectory_28d: (batch, 28) 线性插值轨迹
        """
        alf_start = action_4d[:, 0:1]  # (batch, 1)
        alf_end = action_4d[:, 1:2]    # (batch, 1)
        out_start = action_4d[:, 2:3]  # (batch, 1)
        out_end = action_4d[:, 3:4]    # (batch, 1)

        t = self.t.unsqueeze(0)  # (1, 14, 1)
        alf_traj = alf_start.unsqueeze(1) + (alf_end - alf_start).unsqueeze(1) * t.squeeze(-1)  # (batch, 14)
        out_traj = out_start.unsqueeze(1) + (out_end - out_start).unsqueeze(1) * t.squeeze(-1)  # (batch, 14)

        # 交织为28维
        trajectory = torch.stack([alf_traj, out_traj], dim=-1)  # (batch, 14, 2)
        trajectory_28d = trajectory.reshape(-1, 28)
        return torch.clamp(trajectory_28d, -1, 1)
