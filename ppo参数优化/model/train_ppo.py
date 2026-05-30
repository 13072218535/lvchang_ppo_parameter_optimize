"""
MPD-PPO训练脚本
在仿真环境中训练MPD-PPO算法
"""
import os
import json
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from environment import VoltageControlEnv, load_predictor_model
from ppo import MPDPPO
from config import *

# 中文字体 + 配色（与visualize_ppo.py一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'reward': '#1A73E8',
    'R_acc': '#109618',
    'P_smooth': '#DC3912',
    'P_bound': '#FF9900',
    'actor': '#3366CC',
    'critic': '#888888',
    'smoothed': '#000000',
}

VIS_DIR = os.path.join(OUTPUT_DIR, 'visualization')
os.makedirs(VIS_DIR, exist_ok=True)


def prepare_raw_features(df):
    """
    准备原始特征（仅HIGH_CORR_FEATURES，不做统计/衍生特征工程）。
    优势：避免仿真中统计特征反馈回路，LSTM自行学习时序模式。
    参数:
        df: 原始数据框
    返回:
        df: 处理后数据框
        feature_cols: 特征列名列表（仅12个原始高相关性特征）
    """
    df = df.copy()
    feature_cols = HIGH_CORR_FEATURES.copy()

    # 填充缺失值：按槽号分组前向填充 + 线性插值 + 均值填充
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
            df[col] = df[col].fillna(df[col].mean())

    # 保持原有列顺序：日期、槽号 + 特征列
    required_cols = ['日期', '槽号'] + [c for c in feature_cols if c in df.columns]
    df = df[required_cols]

    return df, feature_cols


def load_data_for_ppo(data_path):
    """
    加载数据用于PPO训练（使用精简原始特征，无统计/衍生特征）
    返回：样本列表、scaler、特征列名列表
    """

    df = pd.read_excel(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)

    # 加载scaler（来自条件预测模型训练，需与predictor特征一致）
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    scaler_path = os.path.join(PROJECT_ROOT, 'model', 'output', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler文件不存在: {scaler_path}\n"
            "请先训练条件预测模型，它会生成scaler.pkl:\n"
            "  步骤1: cd model && 设置 use_conditional=False, 运行 python train.py\n"
            "  步骤2: cd model && 设置 use_conditional=True,  运行 python train.py\n"
            "  步骤3: 再运行本脚本 python train_ppo.py"
        )
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 仅使用原始特征，不做统计/衍生工程（避免仿真失真）
    df, feature_cols = prepare_raw_features(df)
    
    # 获取槽号到索引的映射
    all_pots = sorted(df['槽号'].unique())
    pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}
    
    # 为每个槽号创建训练样本
    samples = []
    
    for pot_id in df['槽号'].unique():
        pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
        
        required_days = INPUT_LEN + OUTPUT_LEN
        if len(pot_data) < required_days:
            continue
        
        pot_features = pot_data[feature_cols].values
        # 目标电压改为设定电压（实际设定），而非历史工作电压
        pot_set_voltages = pot_data['实际设定'].values if '实际设定' in pot_data.columns else pot_data[TARGET].values

        # 创建滑动窗口样本
        for i in range(len(pot_data) - INPUT_LEN - OUTPUT_LEN + 1):
            past_features = pot_features[i:i + INPUT_LEN]
            target_voltage = pot_set_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]

            samples.append({
                'past_features': past_features,
                'target_voltage': target_voltage,
                'pot_id': pot_to_idx[pot_id],
                'pot_num': pot_id
            })
    
    print(f"特征列数量: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")
    
    return samples, scaler, feature_cols


def run_episode(env, ppo_agent, sample, max_steps=MAX_EPISODE_STEPS):
    """
    运行一个episode
    """
    # 重置环境
    state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
    
    total_reward = 0.0
    total_R_acc = 0.0
    total_P_smooth = 0.0
    total_P_bound = 0.0
    episode_data = []

    for step in range(max_steps):
        # 选择动作
        action, log_prob = ppo_agent.select_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)

        # 存储经验
        ppo_agent.store_transition(state, action, reward, log_prob, next_state, done)

        # 更新状态
        state = next_state
        total_reward += reward
        total_R_acc += info.get('R_acc', 0)
        total_P_smooth += info.get('P_smooth', 0)
        total_P_bound += info.get('P_bound', 0)

        # 记录数据
        episode_data.append(info)

        if done:
            break

    reward_components = {
        'R_acc': total_R_acc,
        'P_smooth': total_P_smooth,
        'P_bound': total_P_bound,
    }
    return total_reward, episode_data, reward_components


def moving_average(data, window=10):
    """简单滑动窗口平均，窗口不足时回退到较短的窗口"""
    if len(data) == 0:
        return np.array([])
    arr = np.array(data, dtype=np.float64)
    effective_window = min(window, len(arr))
    if effective_window <= 1:
        return arr
    kernel = np.ones(effective_window) / effective_window
    smoothed = np.convolve(arr, kernel, mode='valid')
    # 前面effective_window-1个点用原始值填充，避免前段空白
    result = np.concatenate([arr[:effective_window - 1], smoothed])
    return result


def plot_training_curves(metrics, save_dir):
    """
    绘制PPO训练过程中的奖励与损失曲线。

    metrics: dict，包含:
        rewards, r_acc, p_smooth, p_bound, actor_losses, critic_losses
    """
    epochs = np.arange(1, len(metrics['rewards']) + 1)
    n = len(epochs)
    if n == 0:
        print("无训练数据，跳过绘图。")
        return

    # 平滑曲线（窗口=PPO_EPOCHS//50，至少5）
    window = max(5, n // 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ===== 面板1: 总Reward =====
    ax = axes[0, 0]
    ax.plot(epochs, metrics['rewards'], alpha=0.2, color=COLORS['reward'],
            linewidth=0.8, label='原始值')
    ax.plot(epochs, moving_average(metrics['rewards'], window),
            color=COLORS['reward'], linewidth=2.0, label=f'平滑 (窗口={window})')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_title('MPD-PPO训练 — 总奖励曲线', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    # ===== 面板2: Reward分量 =====
    ax = axes[0, 1]
    ax.plot(epochs, moving_average(metrics['r_acc'], window),
            color=COLORS['R_acc'], linewidth=1.5, label='R_acc (精度)')
    ax.plot(epochs, moving_average(metrics['p_smooth'], window),
            color=COLORS['P_smooth'], linewidth=1.5, label='P_smooth (平滑)')
    ax.plot(epochs, moving_average(metrics['p_bound'], window),
            color=COLORS['P_bound'], linewidth=1.5, label='P_bound (边界)')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward 分量', fontsize=12, fontweight='bold')
    ax.set_title('奖励函数分量分解 (平滑)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    # ===== 面板3: Actor Loss =====
    ax = axes[1, 0]
    ax.plot(epochs, metrics['actor_losses'], alpha=0.25, color=COLORS['actor'],
            linewidth=0.8, label='原始值')
    smoothed_al = moving_average(metrics['actor_losses'], window)
    ax.plot(epochs, smoothed_al, color=COLORS['actor'], linewidth=2.0,
            label=f'平滑 (窗口={window})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actor Loss', fontsize=12, fontweight='bold')
    ax.set_title('Actor Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    # ===== 面板4: Critic Loss =====
    ax = axes[1, 1]
    ax.plot(epochs, metrics['critic_losses'], alpha=0.25, color=COLORS['critic'],
            linewidth=0.8, label='原始值')
    smoothed_cl = moving_average(metrics['critic_losses'], window)
    ax.plot(epochs, smoothed_cl, color=COLORS['critic'], linewidth=2.0,
            label=f'平滑 (窗口={window})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Critic Loss', fontsize=12, fontweight='bold')
    ax.set_title('Critic Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.suptitle('MPD-PPO训练曲线总览', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    png_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n训练曲线图已保存: {png_path}")


def save_metrics_json(metrics, save_dir):
    """保存训练指标为JSON，便于后续分析"""
    # 转成Python原生类型
    serializable = {}
    for k, v in metrics.items():
        serializable[k] = [float(x) for x in v]
    json_path = os.path.join(save_dir, 'training_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"训练指标已保存: {json_path}")


def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("=" * 60) 
    print("MPD-PPO 训练")
    print("=" * 60)
    
    # 1. 加载训练数据（使用完整特征工程）
    print("\n1. 加载训练数据...")
    train_samples, scaler, feature_cols = load_data_for_ppo(DATA_PATH)
    print(f"训练样本数量: {len(train_samples)}")
    
    # 2. 加载条件预测模型
    print("\n2. 加载条件预测模型...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    model_path = os.path.join(PROJECT_ROOT, 'model', 'output', 'best_conditional_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'model', 'output', 'final_conditional_model.pth')
    print(f"模型路径: {model_path}")
    
    # 获取特征数量
    num_features = len(feature_cols)
    
    # 加载预测器
    predictor = load_predictor_model(model_path, len(TRAIN_POTS) + len(VAL_POTS) + len(TEST_POTS), 
                                     num_features, device)
    print("条件预测模型加载完成")
    
    # 3. 创建仿真环境（传入特征列信息）
    print("\n3. 创建仿真环境...")
    env = VoltageControlEnv(predictor, scaler, feature_cols, device)
    input_dim = len(feature_cols)
    num_pots = len(TRAIN_POTS) + len(VAL_POTS) + len(TEST_POTS)
    state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 4  # +2 last_action +1 cum_err +1 pot_id
    print(f"输入特征维度: {input_dim}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {env.get_action_dim()}")
    print(f"槽数量: {num_pots}")
    
    # 4. 初始化PPO代理（使用轨迹维度 ACTION_TRAJECTORY_DIM=28）
    print("\n4. 初始化PPO代理...")
    print(f"   动作空间: {ACTION_TRAJECTORY_DIM}维 (14天×2动作轨迹)")
    ppo_agent = MPDPPO(input_dim, num_pots, ACTION_TRAJECTORY_DIM, device)
    print("PPO代理初始化完成")

    # 5. 训练循环
    print("\n5. 开始训练...")
    print("=" * 60)

    best_reward = float('-inf')
    total_steps = 0

    # 初始化指标跟踪列表
    metrics = {
        'rewards': [],
        'r_acc': [],
        'p_smooth': [],
        'p_bound': [],
        'actor_losses': [],
        'critic_losses': [],
    }

    for epoch in range(PPO_EPOCHS):
        epoch_reward = 0.0
        epoch_R_acc = 0.0
        epoch_P_smooth = 0.0
        epoch_P_bound = 0.0
        num_episodes = 0

        while total_steps < PPO_STEPS_PER_UPDATE:
            sample = train_samples[np.random.randint(len(train_samples))]

            reward, episode_data, reward_components = run_episode(env, ppo_agent, sample)

            epoch_reward += reward
            epoch_R_acc += reward_components['R_acc']
            epoch_P_smooth += reward_components['P_smooth']
            epoch_P_bound += reward_components['P_bound']
            num_episodes += 1
            total_steps += len(episode_data)

        # 更新PPO网络
        actor_loss, critic_loss = ppo_agent.update()

        # 计算平均奖励
        avg_reward = epoch_reward / num_episodes if num_episodes > 0 else 0
        avg_R_acc = epoch_R_acc / num_episodes if num_episodes > 0 else 0
        avg_P_smooth = epoch_P_smooth / num_episodes if num_episodes > 0 else 0
        avg_P_bound = epoch_P_bound / num_episodes if num_episodes > 0 else 0

        # 记录指标
        metrics['rewards'].append(avg_reward)
        metrics['r_acc'].append(avg_R_acc)
        metrics['p_smooth'].append(avg_P_smooth)
        metrics['p_bound'].append(avg_P_bound)
        metrics['actor_losses'].append(actor_loss)
        metrics['critic_losses'].append(critic_loss)

        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            ppo_agent.save_model(os.path.join(OUTPUT_DIR, 'best_ppo_model.pth'))

        # 打印日志
        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:04d}/{PPO_EPOCHS}] "
                  f"Reward: {avg_reward:.2f} (R_acc={avg_R_acc:.2f} "
                  f"P_sm={avg_P_smooth:.2f} P_bd={avg_P_bound:.2f}) "
                  f"Best: {best_reward:.2f} "
                  f"Actor: {actor_loss:.5f} Critic: {critic_loss:.4f}")

        # 重置步数计数
        total_steps = 0

    print("\n训练完成!")
    print(f"最佳平均奖励: {best_reward:.4f}")

    # 保存最终模型
    ppo_agent.save_model(os.path.join(OUTPUT_DIR, 'final_ppo_model.pth'))
    print(f"最终模型已保存至: {os.path.join(OUTPUT_DIR, 'final_ppo_model.pth')}")

    # 6. 生成训练曲线
    print("\n6. 生成训练曲线...")
    try:
        save_metrics_json(metrics, OUTPUT_DIR)
        plot_training_curves(metrics, VIS_DIR)
    except Exception as e:
        print(f"警告: 训练曲线生成失败: {e}")


if __name__ == '__main__':
    main()