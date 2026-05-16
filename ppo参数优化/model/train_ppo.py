"""
MPD-PPO训练脚本
在仿真环境中训练MPD-PPO算法
"""
import os
import numpy as np
import torch
import pickle
import pandas as pd

from environment import VoltageControlEnv, load_predictor_model
from ppo import MPDPPO
from config import *


def create_features_for_ppo(df):
    """
    为PPO训练创建特征（与data_processor.py一致）
    参数:
        df: 原始数据框
    返回:
        df: 添加特征后的数据框
        feature_cols: 特征列名列表
    """
    df = df.copy()
    feature_cols = HIGH_CORR_FEATURES.copy()
    
    # 为每个高相关性特征创建统计特征和差分特征（不包括目标变量）
    for col in HIGH_CORR_FEATURES:
        if col == TARGET:
            continue
            
        # 3天滑动窗口统计特征
        df[f'{col}_mean_3d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{col}_std_3d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
        
        # 7天滑动窗口统计特征
        df[f'{col}_mean_7d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df[f'{col}_std_7d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
        
        # 1阶差分
        df[f'{col}_diff_1'] = df.groupby('槽号')[col].transform(
            lambda x: x.diff().fillna(0))
        
        # 7阶差分
        df[f'{col}_diff_7'] = df.groupby('槽号')[col].transform(
            lambda x: x.diff(7).fillna(0))
        
        # 添加新特征列名
        feature_cols.extend([
            f'{col}_mean_3d', f'{col}_std_3d',
            f'{col}_mean_7d', f'{col}_std_7d',
            f'{col}_diff_1', f'{col}_diff_7'
        ])
    
    # 为目标变量创建统计特征（用于输入序列）
    df[f'{TARGET}_mean_3d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    df[f'{TARGET}_std_3d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    df[f'{TARGET}_mean_7d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df[f'{TARGET}_std_7d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
    df[f'{TARGET}_diff_1'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.diff().fillna(0))
    df[f'{TARGET}_diff_7'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.diff(7).fillna(0))
    
    feature_cols.extend([
        f'{TARGET}_mean_3d', f'{TARGET}_std_3d',
        f'{TARGET}_mean_7d', f'{TARGET}_std_7d',
        f'{TARGET}_diff_1', f'{TARGET}_diff_7'
    ])
    
    # 衍生特征
    if '工作平均' in df.columns and '电压设定' in df.columns:
        df['电压偏差'] = df['工作平均'] - df['电压设定']
        feature_cols.append('电压偏差')
    
    if '铝水平' in df.columns and '电解质水平' in df.columns:
        df['铝电解比例'] = df['铝水平'] / (df['电解质水平'] + 1e-8)
        feature_cols.append('铝电解比例')
    
    # 槽龄相关特征（非线性处理）
    if '槽龄' in df.columns:
        df['槽龄_log'] = np.log1p(df['槽龄'])
        df['槽龄_squared'] = df['槽龄'] ** 2
        feature_cols.extend(['槽龄_log', '槽龄_squared'])
    
    # 填充任何剩余的NaN值
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    return df, feature_cols


def load_data_for_ppo(data_path):
    """
    加载数据用于PPO训练（使用完整特征工程）
    返回：样本列表、scaler、特征列名列表
    """
    df = pd.read_excel(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)
    
    # 加载scaler（来自条件预测模型训练）
    scaler_path = 'e:/ClaudeCodeWorkplace/2026-5-12-参数优化/model/output/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 创建完整特征（与data_processor.py一致）
    df, feature_cols = create_features_for_ppo(df)
    
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
    model_path = 'e:/ClaudeCodeWorkplace/2026-5-12-参数优化/model/output/best_conditional_model.pth'
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
    state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 1
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


if __name__ == '__main__':
    main()