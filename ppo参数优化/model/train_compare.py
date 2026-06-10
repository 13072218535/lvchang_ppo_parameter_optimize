"""
多算法对比统一训练脚本
支持 --algo mpd_ppo|vanilla_ppo|a2c|ddpg|td3|sac
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
import time
import warnings
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='RNN module weights are not part of single contiguous chunk')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import *
from environment import VoltageControlEnv, load_predictor_model
from algorithms import create_algorithm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'reward': '#1A73E8', 'R_acc': '#109618', 'P_smooth': '#DC3912',
    'P_bound': '#FF9900', 'P_unc': '#990099', 'actor': '#3366CC', 'critic': '#888888',
}


def prepare_raw_features(df):
    df = df.copy()
    feature_cols = HIGH_CORR_FEATURES.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
            df[col] = df[col].fillna(df[col].mean())
    required_cols = ['日期', '槽号'] + [c for c in feature_cols if c in df.columns]
    df = df[required_cols]
    return df, feature_cols


def load_data_for_ppo(data_path):
    df = pd.read_excel(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    scaler_path = os.path.join(PROJECT_ROOT, 'model', 'output', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler文件不存在: {scaler_path}\n请先训练条件预测模型。")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    df, feature_cols = prepare_raw_features(df)
    all_pots = sorted(df['槽号'].unique())
    pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

    samples = []
    for pot_id in df['槽号'].unique():
        pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
        required_days = INPUT_LEN + OUTPUT_LEN
        if len(pot_data) < required_days:
            continue
        pot_features = pot_data[feature_cols].values
        pot_set_voltages = pot_data['实际设定'].values if '实际设定' in pot_data.columns else pot_data[TARGET].values
        for i in range(len(pot_data) - INPUT_LEN - OUTPUT_LEN + 1):
            samples.append({
                'past_features': pot_features[i:i + INPUT_LEN],
                'target_voltage': pot_set_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
                'pot_id': pot_to_idx[pot_id], 'pot_num': pot_id
            })
    return samples, scaler, feature_cols


def run_episode_on_policy(env, algo, sample):
    """On-policy episode（MPD-PPO / Vanilla PPO / A2C）"""
    state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
    total_reward = 0.0
    total_R_acc = 0.0
    total_P_smooth = 0.0
    total_P_bound = 0.0
    total_P_unc = 0.0
    episode_data = []

    for _ in range(MAX_EPISODE_STEPS):
        action, log_prob = algo.select_action(state)
        next_state, reward, done, info = env.step(action)
        algo.store_transition(state, action, reward, log_prob, next_state, done)
        state = next_state
        total_reward += reward
        total_R_acc += info.get('R_acc', 0)
        total_P_smooth += info.get('P_smooth', 0)
        total_P_bound += info.get('P_bound', 0)
        total_P_unc += info.get('P_unc', 0)
        episode_data.append(info)
        if done:
            break

    return total_reward, episode_data, {
        'R_acc': total_R_acc, 'P_smooth': total_P_smooth,
        'P_bound': total_P_bound, 'P_unc': total_P_unc
    }


def run_episode_off_policy(env, algo, sample):
    """Off-policy episode（DDPG / TD3 / SAC）— 每步更新"""
    state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
    total_reward = 0.0
    total_R_acc = 0.0
    total_P_smooth = 0.0
    total_P_bound = 0.0
    total_P_unc = 0.0
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    update_count = 0
    episode_data = []

    for _ in range(MAX_EPISODE_STEPS):
        action = algo.select_action(state)
        next_state, reward, done, info = env.step(action)
        algo.store_transition(state, action, reward, None, next_state, done)

        # Off-policy每步更新
        al, cl = algo.update()
        if al != 0.0 or cl != 0.0:
            actor_loss_sum += al
            critic_loss_sum += cl
            update_count += 1

        state = next_state
        total_reward += reward
        total_R_acc += info.get('R_acc', 0)
        total_P_smooth += info.get('P_smooth', 0)
        total_P_bound += info.get('P_bound', 0)
        total_P_unc += info.get('P_unc', 0)
        episode_data.append(info)
        if done:
            break

    avg_al = actor_loss_sum / max(update_count, 1)
    avg_cl = critic_loss_sum / max(update_count, 1)
    return total_reward, episode_data, {
        'R_acc': total_R_acc, 'P_smooth': total_P_smooth,
        'P_bound': total_P_bound, 'P_unc': total_P_unc
    }, avg_al, avg_cl


def moving_average(data, window=10):
    if len(data) == 0: return np.array([])
    arr = np.array(data, dtype=np.float64)
    ew = min(window, len(arr))
    if ew <= 1: return arr
    kernel = np.ones(ew) / ew
    smoothed = np.convolve(arr, kernel, mode='valid')
    return np.concatenate([arr[:ew - 1], smoothed])


def plot_training_curves(metrics, save_path):
    epochs = np.arange(1, len(metrics['rewards']) + 1)
    n = len(epochs)
    if n == 0: return
    window = max(5, n // 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    ax.plot(epochs, metrics['rewards'], alpha=0.2, color=COLORS['reward'], linewidth=0.8, label='Raw')
    ax.plot(epochs, moving_average(metrics['rewards'], window),
            color=COLORS['reward'], linewidth=2.0, label=f'Smooth(win={window})')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Reward')
    ax.set_title('Training Reward'); ax.legend(); ax.grid(True, alpha=0.2, linestyle='--')

    ax = axes[0, 1]
    ax.plot(epochs, moving_average(metrics['r_acc'], window), color=COLORS['R_acc'], linewidth=1.5, label='R_acc')
    ax.plot(epochs, moving_average(metrics['p_smooth'], window), color=COLORS['P_smooth'], linewidth=1.5, label='P_smooth')
    ax.plot(epochs, moving_average(metrics['p_bound'], window), color=COLORS['P_bound'], linewidth=1.5, label='P_bound')
    if metrics.get('p_unc') and any(v != 0 for v in metrics['p_unc']):
        ax.plot(epochs, moving_average(metrics['p_unc'], window), color=COLORS['P_unc'], linewidth=1.5, label='P_unc')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Components')
    ax.set_title('Reward Components (Smoothed)'); ax.legend(); ax.grid(True, alpha=0.2, linestyle='--')

    ax = axes[1, 0]
    ax.plot(epochs, metrics['actor_losses'], alpha=0.25, color=COLORS['actor'], linewidth=0.8, label='Raw')
    ax.plot(epochs, moving_average(metrics['actor_losses'], window),
            color=COLORS['actor'], linewidth=2.0, label=f'Smooth')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Actor Loss'); ax.set_title('Actor Loss')
    ax.legend(); ax.grid(True, alpha=0.2, linestyle='--')

    ax = axes[1, 1]
    ax.plot(epochs, metrics['critic_losses'], alpha=0.25, color=COLORS['critic'], linewidth=0.8, label='Raw')
    ax.plot(epochs, moving_average(metrics['critic_losses'], window),
            color=COLORS['critic'], linewidth=2.0, label=f'Smooth')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Critic Loss'); ax.set_title('Critic Loss')
    ax.legend(); ax.grid(True, alpha=0.2, linestyle='--')

    fig.suptitle('PPO Training Curves', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved: {save_path}")


def train_on_policy(algo, algo_name, env, train_samples, algo_dir, epochs=PPO_EPOCHS):
    """On-policy训练循环"""
    best_reward = float('-inf')
    metrics = {'rewards': [], 'r_acc': [], 'p_smooth': [], 'p_bound': [],
               'p_unc': [], 'actor_losses': [], 'critic_losses': []}
    total_steps = 0

    for epoch in range(epochs):
        # TAA-PPO: 通知当前epoch以驱动预热/自适应调度
        if hasattr(algo, 'set_epoch'):
            algo.set_epoch(epoch)

        epoch_reward = 0.0; epoch_R = 0.0; epoch_PS = 0.0; epoch_PB = 0.0; epoch_PU = 0.0
        num_episodes = 0

        while total_steps < PPO_STEPS_PER_UPDATE:
            sample = train_samples[np.random.randint(len(train_samples))]
            reward, episode_data, rc = run_episode_on_policy(env, algo, sample)
            epoch_reward += reward
            epoch_R += rc['R_acc']; epoch_PS += rc['P_smooth']  # noqa: E702
            epoch_PB += rc['P_bound']; epoch_PU += rc['P_unc']  # noqa: E702
            num_episodes += 1
            total_steps += len(episode_data)

        actor_loss, critic_loss = algo.update()

        avg_reward = epoch_reward / max(num_episodes, 1)
        metrics['rewards'].append(avg_reward)
        metrics['r_acc'].append(epoch_R / max(num_episodes, 1))
        metrics['p_smooth'].append(epoch_PS / max(num_episodes, 1))
        metrics['p_bound'].append(epoch_PB / max(num_episodes, 1))
        metrics['p_unc'].append(epoch_PU / max(num_episodes, 1))
        metrics['actor_losses'].append(actor_loss)
        metrics['critic_losses'].append(critic_loss)

        if avg_reward > best_reward:
            best_reward = avg_reward
            algo.save_model(os.path.join(algo_dir, 'best_model.pth'))

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            unc_str = f" Pu={metrics['p_unc'][-1]:.2f}" if metrics['p_unc'][-1] != 0 else ""
            print(f"Epoch [{epoch + 1:04d}/{epochs}] "
                  f"Reward: {avg_reward:.2f} (R={metrics['r_acc'][-1]:.2f} "
                  f"Ps={metrics['p_smooth'][-1]:.2f} Pb={metrics['p_bound'][-1]:.2f}{unc_str}) "
                  f"Best: {best_reward:.2f} "
                  f"AL: {actor_loss:.5f} CL: {critic_loss:.4f}")

        total_steps = 0

    return metrics, best_reward


def train_off_policy(algo, algo_name, env, train_samples, algo_dir, epochs=PPO_EPOCHS):
    """Off-policy训练循环"""
    best_reward = float('-inf')
    metrics = {'rewards': [], 'r_acc': [], 'p_smooth': [], 'p_bound': [],
               'p_unc': [], 'actor_losses': [], 'critic_losses': []}
    steps_per_epoch = PPO_STEPS_PER_UPDATE
    warmup_steps = 256  # replay buffer填充阈值

    # 预热阶段：快速填充replay buffer
    print(f"  Warming up replay buffer ({warmup_steps} steps)...")
    t_warmup = time.time()
    warmup_done = 0
    while len(algo.replay_buffer) < warmup_steps:
        sample = train_samples[np.random.randint(len(train_samples))]
        state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
        for _ in range(MAX_EPISODE_STEPS):
            if len(algo.replay_buffer) >= warmup_steps: break
            action = algo.select_action(state)
            next_state, reward, done, info = env.step(action)
            algo.store_transition(state, action, reward, None, next_state, done)
            state = next_state
            warmup_done += 1
            if done: break
    print(f"  Warmup complete ({len(algo.replay_buffer)} transitions, {time.time()-t_warmup:.0f}s)")

    for epoch in range(epochs):
        epoch_reward = 0.0; epoch_R = 0.0; epoch_PS = 0.0; epoch_PB = 0.0; epoch_PU = 0.0
        epoch_al = 0.0; epoch_cl = 0.0
        num_episodes = 0; total_steps = 0

        # 改进3：课程学习 — 平滑惩罚从低到高线性增长
        progress = min(1.0, epoch / max(epochs * 0.7, 1))
        cur_weight = 1.0 + (REWARD_SMOOTH_VIOLATION_WEIGHT - 1.0) * progress
        if hasattr(env, 'set_smoothness_weight'):
            env.set_smoothness_weight(cur_weight)

        while total_steps < steps_per_epoch:
            sample = train_samples[np.random.randint(len(train_samples))]
            reward, episode_data, rc, al, cl = run_episode_off_policy(env, algo, sample)
            epoch_reward += reward
            epoch_R += rc['R_acc']; epoch_PS += rc['P_smooth']  # noqa: E702
            epoch_PB += rc['P_bound']; epoch_PU += rc['P_unc']  # noqa: E702
            epoch_al += al; epoch_cl += cl
            num_episodes += 1
            total_steps += len(episode_data)

        avg_reward = epoch_reward / max(num_episodes, 1)
        metrics['rewards'].append(avg_reward)
        metrics['r_acc'].append(epoch_R / max(num_episodes, 1))
        metrics['p_smooth'].append(epoch_PS / max(num_episodes, 1))
        metrics['p_bound'].append(epoch_PB / max(num_episodes, 1))
        metrics['p_unc'].append(epoch_PU / max(num_episodes, 1))
        metrics['actor_losses'].append(epoch_al / max(num_episodes, 1))
        metrics['critic_losses'].append(epoch_cl / max(num_episodes, 1))

        if avg_reward > best_reward:
            best_reward = avg_reward
            algo.save_model(os.path.join(algo_dir, 'best_model.pth'))

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            unc_str = f" Pu={metrics['p_unc'][-1]:.2f}" if metrics['p_unc'][-1] != 0 else ""
            print(f"Epoch [{epoch + 1:04d}/{epochs}] "
                  f"Reward: {avg_reward:.2f} (R={metrics['r_acc'][-1]:.2f} "
                  f"Ps={metrics['p_smooth'][-1]:.2f} Pb={metrics['p_bound'][-1]:.2f}{unc_str}) "
                  f"Best: {best_reward:.2f} "
                  f"AL: {metrics['actor_losses'][-1]:.5f} CL: {metrics['critic_losses'][-1]:.4f}")

    return metrics, best_reward


def main():
    parser = argparse.ArgumentParser(description='Multi-Algorithm PPO Training Comparison')
    parser.add_argument('--algo', type=str, default='mpd_ppo',
                        choices=['mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo', 'taa_ppo_4d', 'ddpg', 'td3', 'sac'],
                        help='Algorithm to train')
    parser.add_argument('--epochs', type=int, default=PPO_EPOCHS, help='Training epochs')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    algo_name = args.algo
    print("=" * 60)
    print(f"Training: {algo_name.upper()}（{'On-policy' if algo_name in ['mpd_ppo','vanilla_ppo','a2c','taa_ppo','taa_ppo_4d'] else 'Off-policy'}）")
    print(f"Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("=" * 60)

    # 输出目录
    algo_dir = os.path.join(OUTPUT_DIR, algo_name)
    vis_dir = os.path.join(algo_dir, 'visualization')
    os.makedirs(algo_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 加载数据
    print("\n1. Loading training data...")
    train_samples, scaler, feature_cols = load_data_for_ppo(DATA_PATH)
    print(f"   Samples: {len(train_samples)}, Features: {len(feature_cols)}")

    # 2. 加载预测模型（自动检测ensemble）
    print("\n2. Loading condition predictor...")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'model', 'output')

    # 检测ensemble checkpoints
    import glob as glob_module
    ensemble_pattern = os.path.join(MODEL_OUTPUT_DIR, 'best_conditional_seed*.pth')
    ensemble_paths = sorted(glob_module.glob(ensemble_pattern))

    num_features = len(feature_cols)
    num_pots = len(TRAIN_POTS) + len(VAL_POTS) + len(TEST_POTS)

    ensemble_is_ensemble = False
    if len(ensemble_paths) >= UNC_NUM_ENSEMBLE:
        print(f"   Found {len(ensemble_paths)} ensemble checkpoints, using ensemble predictor")
        ensemble_paths = ensemble_paths[:UNC_NUM_ENSEMBLE]
        predictor = load_predictor_model(
            None, num_pots, num_features, device,
            ensemble=True, ensemble_checkpoint_paths=ensemble_paths
        )
        ensemble_is_ensemble = True
    else:
        print(f"   Found {len(ensemble_paths)} ensemble checkpoints (< {UNC_NUM_ENSEMBLE}), using single predictor")
        model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_conditional_model_augmented.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_conditional_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'final_conditional_model.pth')
        print(f"   Model: {model_path}")
        predictor = load_predictor_model(model_path, num_pots, num_features, device)
    print("   Predictor loaded.")

    # 3. 加载不确定性阈值（如果使用ensemble）
    unc_threshold = UNC_THRESHOLD
    if ensemble_is_ensemble and UNC_USE_THRESHOLD and unc_threshold is None:
        threshold_path = os.path.join(MODEL_OUTPUT_DIR, 'uncertainty_threshold.json')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
            unc_threshold = threshold_data.get('P95', None)
            print(f"   Uncertainty threshold loaded: P95={unc_threshold:.6f}" if unc_threshold else
                  f"   WARNING: No P95 value in threshold file")

    # 4. 创建环境
    print("\n4. Creating environment...")
    uncertainty_config = {
        'lambda': UNC_LAMBDA,
        'use_threshold': UNC_USE_THRESHOLD,
        'threshold': unc_threshold,
    }
    env = VoltageControlEnv(predictor, scaler, feature_cols, device,
                            uncertainty_config=uncertainty_config,
                            ensemble_is_ensemble=ensemble_is_ensemble)
    input_dim = len(feature_cols)
    state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 4  # +2 last_action +1 cum_err +1 pot_id
    print(f"   State dim: {state_dim}, Action dim: {ACTION_TRAJECTORY_DIM}")

    # 5. 创建算法
    print(f"\n5. Initializing {algo_name.upper()}...")
    algo = create_algorithm(algo_name, input_dim, num_pots, ACTION_TRAJECTORY_DIM, device)
    print(f"   Algorithm initialized.")

    # 6. 训练
    t_start = time.time()

    # TAA-PPO-4D: 4维架构无法独立控制每天动作，降低边界惩罚
    if algo_name == 'taa_ppo_4d' and hasattr(env, 'set_bound_penalty_weight'):
        env.set_bound_penalty_weight(TAA4D_BOUND_PENALTY_WEIGHT)
        print(f"   TAA-PPO-4D: bound penalty weight set to {TAA4D_BOUND_PENALTY_WEIGHT}")

    if algo_name in ['mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo', 'taa_ppo_4d']:
        metrics, best_r = train_on_policy(algo, algo_name, env, train_samples, algo_dir, args.epochs)
    else:
        metrics, best_r = train_off_policy(algo, algo_name, env, train_samples, algo_dir, args.epochs)
    t_elapsed = time.time() - t_start

    print(f"\nTraining complete! Best reward: {best_r:.4f}, Time: {t_elapsed:.1f}s")

    # 7. 保存模型 & 指标
    algo.save_model(os.path.join(algo_dir, 'final_model.pth'))
    with open(os.path.join(algo_dir, 'training_metrics.json'), 'w', encoding='utf-8') as f:
        serializable = {k: [float(x) for x in v] for k, v in metrics.items()}
        serializable['training_time_s'] = t_elapsed
        serializable['algo_name'] = algo_name
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    # 8. 生成训练曲线
    print("\n7. Generating training curves...")
    try:
        plot_training_curves(metrics, os.path.join(vis_dir, 'training_curves.png'))
    except Exception as e:
        print(f"Warning: curve generation failed: {e}")

    print(f"\nAll outputs saved to: {algo_dir}")
    print(f"  - best_model.pth / final_model.pth")
    print(f"  - training_metrics.json")
    print(f"  - visualization/training_curves.png")


if __name__ == '__main__':
    main()
