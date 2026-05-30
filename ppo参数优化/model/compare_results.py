"""
多算法训练结果对比脚本
读取各算法的 training_metrics.json，生成训练曲线叠加对比图 + 收敛速度分析
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 Windows OpenMP 冲突

import sys
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from config import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ALGO_COLORS = {
    'mpd_ppo': '#1A73E8', 'vanilla_ppo': '#DC3912', 'a2c': '#888888',
    'ddpg': '#109618', 'td3': '#FF9900', 'sac': '#990099',
    'taa_ppo': '#00BCD4',  # 醒目的青蓝色，与所有现有颜色区分
}
ALGO_NAMES = {
    'mpd_ppo': 'MPD-PPO', 'vanilla_ppo': 'Vanilla PPO', 'a2c': 'A2C',
    'ddpg': 'DDPG', 'td3': 'TD3', 'sac': 'SAC',
    'taa_ppo': 'TAA-PPO',
}
# 线型区分：on-policy实线，off-policy虚线，TAA-PPO加粗突出
ALGO_LINESTYLES = {
    'mpd_ppo': '-', 'vanilla_ppo': '-', 'a2c': '-',
    'ddpg': '--', 'td3': '--', 'sac': '--',
    'taa_ppo': '-',  # 实线
}
ALGO_LINEWIDTHS = {
    'mpd_ppo': 1.5, 'vanilla_ppo': 1.5, 'a2c': 1.5,
    'ddpg': 1.2, 'td3': 1.2, 'sac': 1.2,
    'taa_ppo': 2.8,  # 加粗突出
}


def moving_average(data, window=10):
    if len(data) == 0: return np.array([])
    arr = np.array(data, dtype=np.float64)
    ew = min(window, len(arr))
    if ew <= 1: return arr
    kernel = np.ones(ew) / ew
    smoothed = np.convolve(arr, kernel, mode='valid')
    return np.concatenate([arr[:ew - 1], smoothed])


def load_algo_metrics(algo_name):
    """加载算法训练指标"""
    json_path = os.path.join(OUTPUT_DIR, algo_name, 'training_metrics.json')
    if not os.path.exists(json_path):
        print(f"  Skip {algo_name}: {json_path} not found")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_reward_comparison(all_metrics, save_path, algos):
    """图表1：7算法训练曲线叠加对比 — 差异放大布局"""
    fig, axes = plt.subplots(3, 2, figsize=(20, 17))

    # ── 子图1-1: 总奖励（全尺度） ──
    ax = axes[0, 0]
    for algo_name in algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        epochs = np.arange(1, len(m['rewards']) + 1)
        n = len(epochs)
        window = max(5, n // 50)
        smoothed = moving_average(m['rewards'], window)
        ls = ALGO_LINESTYLES.get(algo_name, '-')
        lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)
        ax.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Reward (Smoothed)')
    ax.set_title('A. Full-Scale Reward Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=4, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')

    # ── 子图1-2: 核心对比 — 偏离Vanilla PPO基线（差距放大） ──
    ax = axes[0, 1]
    # 以Vanilla PPO为基准，计算其他算法的reward差值
    baseline = all_metrics.get('vanilla_ppo')
    if baseline is not None:
        bl_rewards = np.array(baseline['rewards'])
        bl_n = len(bl_rewards)
        bl_window = max(5, bl_n // 50)
        bl_smoothed = moving_average(bl_rewards, bl_window)

        for algo_name in algos:
            m = all_metrics.get(algo_name)
            if m is None: continue
            rewards = np.array(m['rewards'])
            n_epochs = len(rewards)
            window = max(5, n_epochs // 50)
            smoothed = moving_average(rewards, window)
            # 对齐长度（不同算法的epoch数可能略有差异）
            min_len = min(len(smoothed), len(bl_smoothed))
            diff = smoothed[:min_len] - bl_smoothed[:min_len]
            epochs = np.arange(1, min_len + 1)
            ls = ALGO_LINESTYLES.get(algo_name, '-')
            lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)
            # Vanilla PPO自身差值为0（灰色虚线）
            if algo_name == 'vanilla_ppo':
                ax.axhline(y=0, color='#DC3912', linestyle='--', linewidth=1.2, alpha=0.5,
                          label='Vanilla PPO (baseline)')
            else:
                ax.plot(epochs, diff, color=ALGO_COLORS[algo_name], linewidth=lw,
                        linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Δ Reward vs Vanilla PPO')
    ax.set_title('B. Reward Gap vs Vanilla PPO (Differences Amplified)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')

    # ── 子图2-1: 超紧密缩放（735-752，仅on-policy） ──
    ax = axes[1, 0]
    focus_algos = ['mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo']
    for algo_name in focus_algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        epochs = np.arange(1, len(m['rewards']) + 1)
        n = len(epochs)
        window = max(5, n // 50)
        smoothed = moving_average(m['rewards'], window)
        lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)
        ax.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                linestyle='-', label=ALGO_NAMES[algo_name], alpha=0.9)
    ax.set_ylim(735, 752)
    ax.axhline(y=748, color='gray', linestyle=':', linewidth=0.6, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Reward')
    ax.set_title('C. Ultra-Zoom (735–752) — On-Policy Fine Ranking', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')

    # ── 子图2-2: R_acc ZOOM ──
    ax = axes[1, 1]
    for algo_name in algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        epochs = np.arange(1, len(m['r_acc']) + 1)
        n = len(epochs)
        window = max(5, n // 50)
        smoothed = moving_average(m['r_acc'], window)
        ls = ALGO_LINESTYLES.get(algo_name, '-')
        lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)
        ax.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.85)
    ax.set_ylim(700, 755)
    ax.set_xlabel('Epoch'); ax.set_ylabel('R_acc (Accuracy)')
    ax.set_title('D. R_acc ZOOM (700–755)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=4, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')

    # ── 子图3-1: P_smooth分量 ──
    ax = axes[2, 0]
    for algo_name in algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        epochs = np.arange(1, len(m['p_smooth']) + 1)
        n = len(epochs)
        window = max(5, n // 50)
        smoothed = moving_average(m['p_smooth'], window)
        ls = ALGO_LINESTYLES.get(algo_name, '-')
        lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)
        ax.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.85)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('P_smooth (Smoothness Penalty)')
    ax.set_title('E. Smoothness Penalty Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=4, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')

    # ── 子图3-2: Actor Loss（双Y轴：左侧on-policy, 右侧off-policy） ──
    ax = axes[2, 1]
    ax_off = ax.twinx()  # 右侧Y轴给off-policy（量级完全不同）

    on_style = dict(colors=[], labels=[], linewidths=[])
    off_style = dict(colors=[], labels=[], linewidths=[])

    on_policy_set = {'mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo'}
    for algo_name in algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        epochs = np.arange(1, len(m['actor_losses']) + 1)
        n = len(epochs)
        window = max(5, n // 50)
        smoothed = moving_average(m['actor_losses'], window)
        ls = ALGO_LINESTYLES.get(algo_name, '-')
        lw = ALGO_LINEWIDTHS.get(algo_name, 1.5)

        if algo_name in on_policy_set:
            ax.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                    linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.9)
            on_style['colors'].append(ALGO_COLORS[algo_name])
            on_style['labels'].append(ALGO_NAMES[algo_name])
            on_style['linewidths'].append(lw)
        else:
            ax_off.plot(epochs, smoothed, color=ALGO_COLORS[algo_name], linewidth=lw,
                        linestyle=ls, label=ALGO_NAMES[algo_name], alpha=0.65)
            off_style['colors'].append(ALGO_COLORS[algo_name])
            off_style['labels'].append(ALGO_NAMES[algo_name])
            off_style['linewidths'].append(lw)

    ax.set_ylim(-0.15, 0.15)
    ax.set_ylabel('On-Policy Actor Loss', color='#333333')
    ax_off.set_ylabel('Off-Policy Actor Loss (-Q value)', color='#888888')
    ax.set_xlabel('Epoch')
    ax.set_title('F. Actor Loss (Dual-Scale: Left=On-Policy, Right=Off-Policy)',
                 fontsize=13, fontweight='bold')

    # 合并图例
    lines_on = [plt.Line2D([0],[0], color=c, linewidth=w, linestyle='-')
                for c, w in zip(on_style['colors'], on_style['linewidths'])]
    lines_off = [plt.Line2D([0],[0], color=c, linewidth=w, linestyle='--')
                 for c, w in zip(off_style['colors'], off_style['linewidths'])]
    ax.legend(lines_on + lines_off, on_style['labels'] + off_style['labels'],
              fontsize=7.5, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.15, linestyle='--')

    fig.suptitle('Multi-Algorithm Training Comparison (7 Algorithms × 200 Epochs)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reward comparison saved: {save_path}")


def plot_convergence_analysis(all_metrics, save_path, algos):
    """图表2：收敛速度分析（首次达标epoch + 最终稳定值 + 训练耗时）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 数据整理
    algo_labels = []
    t90_list = []       # 首次达到90%最终reward的epoch
    final_reward_list = []  # 最后100 epoch平均reward
    final_std_list = []     # 最终稳定性
    train_time_list = []    # 训练耗时
    colors_list = []

    for algo_name in algos:
        m = all_metrics.get(algo_name)
        if m is None: continue
        algo_labels.append(ALGO_NAMES[algo_name])
        colors_list.append(ALGO_COLORS[algo_name])

        rewards = np.array(m['rewards'])
        final_window = min(100, len(rewards))
        final_mean = np.mean(rewards[-final_window:])
        final_std = np.std(rewards[-final_window:])
        final_reward_list.append(final_mean)
        final_std_list.append(final_std)

        # T_90%: 首次达到 final_mean * 0.9 的epoch
        target = final_mean * 0.9
        reached = np.where(rewards >= target)[0]
        t90 = reached[0] + 1 if len(reached) > 0 else len(rewards)
        t90_list.append(t90)

        train_time_list.append(m.get('training_time_s', 0))

    # 子图1: 收敛速度（T_90%）
    ax = axes[0]
    x = np.arange(len(algo_labels))
    bars = ax.bar(x, t90_list, color=colors_list, alpha=0.85, edgecolor='white')
    for i, v in enumerate(t90_list):
        ax.text(i, v + max(t90_list) * 0.02, str(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=10, rotation=15)
    ax.set_ylabel('Epochs to 90% Final Reward', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Speed (lower is better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    # 子图2: 最终收敛值 + 稳定性
    ax = axes[1]
    x = np.arange(len(algo_labels))
    bars = ax.bar(x, final_reward_list, yerr=final_std_list, color=colors_list,
                  alpha=0.85, edgecolor='white', capsize=5, ecolor='#333333')
    for i, (v, s) in enumerate(zip(final_reward_list, final_std_list)):
        ax.text(i, v + s + max(final_reward_list) * 0.02,
                f'{v:.1f}±{s:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=10, rotation=15)
    ax.set_ylabel('Final Reward (±1σ)', fontsize=12, fontweight='bold')
    ax.set_title('Final Reward Level (higher is better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    # 子图3: 训练耗时
    ax = axes[2]
    bars = ax.bar(x, train_time_list, color=colors_list, alpha=0.85, edgecolor='white')
    for i, v in enumerate(train_time_list):
        ax.text(i, v + max(train_time_list) * 0.02, f'{v:.0f}s', ha='center',
                fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=10, rotation=15)
    ax.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time (lower is better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    fig.suptitle('Convergence Speed & Stability Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Convergence analysis saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['mpd_ppo', 'vanilla_ppo', 'a2c', 'ddpg', 'td3', 'sac', 'taa_ppo'])
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output or os.path.join(OUTPUT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Algorithm Training Results Comparison")
    print("=" * 60)

    # 1. 加载所有算法指标
    print("\n1. Loading training metrics...")
    all_metrics = {}
    avail_algos = []
    for algo_name in args.algos:
        m = load_algo_metrics(algo_name)
        if m is not None:
            all_metrics[algo_name] = m
            avail_algos.append(algo_name)
            epochs = len(m['rewards'])
            t = m.get('training_time_s', 0)
            print(f"   {ALGO_NAMES[algo_name]}: {epochs} epochs, {t:.0f}s, "
                  f"final reward={np.mean(m['rewards'][-min(100, len(m['rewards'])):]):.2f}")

    if len(avail_algos) < 2:
        print("\nError: Need at least 2 trained algorithms for comparison.")
        print("Train algorithms first: python train_compare.py --algo <name> --epochs 200")
        return

    # 2. 生成奖励曲线对比图
    print("\n2. Generating reward comparison...")
    plot_reward_comparison(all_metrics, os.path.join(output_dir, 'reward_comparison.png'), avail_algos)

    # 3. 生成收敛分析图
    print("\n3. Generating convergence analysis...")
    plot_convergence_analysis(all_metrics, os.path.join(output_dir, 'convergence_analysis.png'), avail_algos)

    # 4. 打印汇总表格
    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 70)
    print(f"{'Algorithm':<16} {'T_90%':>6} {'Final Reward':>14} {'Stability':>10} {'Time':>8}")
    print("-" * 70)
    for algo_name in avail_algos:
        m = all_metrics[algo_name]
        rewards = np.array(m['rewards'])
        fw = min(100, len(rewards))
        fm = np.mean(rewards[-fw:])
        fs = np.std(rewards[-fw:])
        target = fm * 0.9
        reached = np.where(rewards >= target)[0]
        t90 = reached[0] + 1 if len(reached) > 0 else len(rewards)
        print(f"{ALGO_NAMES[algo_name]:<16} {t90:>6} {fm:>14.2f} {fs:>10.2f} "
              f"{m.get('training_time_s', 0):>8.0f}s")
    print("-" * 70)
    print(f"\nOutput: {output_dir}")


if __name__ == '__main__':
    main()
