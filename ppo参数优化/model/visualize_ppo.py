"""
PPO参数优化结果可视化脚本
展示通过PPO调整ALF加料量与实际出铝量后，工作平均电压向设定电压基准线逼近的优化过程。
"""
import os
import sys
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import config as ppo_config
ppo_config.DATA_PATH = os.path.join(PROJECT_ROOT, '槽况数据_处理后_v2.xlsx')
ppo_config.OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ppo参数优化', 'model', 'output')

from environment import VoltageControlEnv, load_predictor_model
from ppo import MPDPPO
from config import *

SCALER_PATH = os.path.join(PROJECT_ROOT, 'model', 'output', 'scaler.pkl')
PREDICTOR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'output', 'best_conditional_model.pth')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLOR_VOLTAGE = '#1A73E8'       # 电压曲线 - 蓝色
COLOR_SETPOINT = '#DC3912'      # 设定电压基准线 - 红色
COLOR_ERROR_FILL = '#FF6B6B'    # 误差填充
COLOR_ALF = '#109618'           # ALF - 绿色
COLOR_OUT = '#3366CC'           # 出铝量 - 深蓝
COLOR_MEAN = '#888888'          # 均值线 - 灰色
COLOR_INCREASE = '#DC3912'      # 动作增加 - 红色
COLOR_DECREASE = '#3366CC'      # 动作减少 - 蓝色
COLOR_CONVERGE = '#FF9900'      # 收敛方向标注


def create_features_for_ppo(df):
    """为PPO创建特征（与train_ppo.py完全一致）"""
    df = df.copy()
    feature_cols = HIGH_CORR_FEATURES.copy()

    for col in HIGH_CORR_FEATURES:
        if col == TARGET:
            continue
        df[f'{col}_mean_3d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{col}_std_3d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
        df[f'{col}_mean_7d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df[f'{col}_std_7d'] = df.groupby('槽号')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
        df[f'{col}_diff_1'] = df.groupby('槽号')[col].transform(lambda x: x.diff().fillna(0))
        df[f'{col}_diff_7'] = df.groupby('槽号')[col].transform(lambda x: x.diff(7).fillna(0))
        feature_cols.extend([
            f'{col}_mean_3d', f'{col}_std_3d',
            f'{col}_mean_7d', f'{col}_std_7d',
            f'{col}_diff_1', f'{col}_diff_7'
        ])

    df[f'{TARGET}_mean_3d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    df[f'{TARGET}_std_3d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    df[f'{TARGET}_mean_7d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df[f'{TARGET}_std_7d'] = df.groupby('槽号')[TARGET].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
    df[f'{TARGET}_diff_1'] = df.groupby('槽号')[TARGET].transform(lambda x: x.diff().fillna(0))
    df[f'{TARGET}_diff_7'] = df.groupby('槽号')[TARGET].transform(lambda x: x.diff(7).fillna(0))
    feature_cols.extend([
        f'{TARGET}_mean_3d', f'{TARGET}_std_3d',
        f'{TARGET}_mean_7d', f'{TARGET}_std_7d',
        f'{TARGET}_diff_1', f'{TARGET}_diff_7'
    ])

    if '工作平均' in df.columns and '电压设定' in df.columns:
        df['电压偏差'] = df['工作平均'] - df['电压设定']
        feature_cols.append('电压偏差')
    if '铝水平' in df.columns and '电解质水平' in df.columns:
        df['铝电解比例'] = df['铝水平'] / (df['电解质水平'] + 1e-8)
        feature_cols.append('铝电解比例')
    if '槽龄' in df.columns:
        df['槽龄_log'] = np.log1p(df['槽龄'])
        df['槽龄_squared'] = df['槽龄'] ** 2
        feature_cols.extend(['槽龄_log', '槽龄_squared'])

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    return df, feature_cols


def load_test_data(data_path, test_pots):
    """加载测试槽数据，返回含有历史特征+未来目标+实际电压的样本列表"""
    df = pd.read_excel(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    df, feature_cols = create_features_for_ppo(df)
    all_pots = sorted(df['槽号'].unique())
    pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

    samples = []
    for pot_id in test_pots:
        if pot_id not in df['槽号'].unique():
            continue
        pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
        required_days = INPUT_LEN + OUTPUT_LEN
        if len(pot_data) < required_days:
            continue

        pot_features = pot_data[feature_cols].values
        pot_set_voltages = pot_data['实际设定'].values
        pot_actual_voltages = pot_data['工作平均'].values
        pot_dates = pot_data['日期'].values

        for i in range(len(pot_data) - INPUT_LEN - OUTPUT_LEN + 1):
            samples.append({
                'past_features': pot_features[i:i + INPUT_LEN],
                'target_voltage': pot_set_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
                'actual_voltage': pot_actual_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
                'pot_id': pot_to_idx[pot_id],
                'pot_num': pot_id,
                'dates_past': pot_dates[i:i + INPUT_LEN],
                'dates_future': pot_dates[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
            })
    return samples, scaler, feature_cols


def run_evaluation_episode(env, ppo_agent, sample):
    """运行完整评估episode，记录每步电压、动作、误差，检测电压异常"""
    state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])

    history = {
        'voltage_pred': [],
        'voltage_set': [],
        'voltage_actual': [],
        'alf_action': [],
        'out_action': [],
        'voltage_error': [],
    }
    voltage_anomaly = False

    for step in range(MAX_EPISODE_STEPS):
        action, _ = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        v_pred = info['voltage_pred']
        v_set = info['target_voltage_set'] if info['target_voltage_set'] is not None else np.nan

        # 电压异常检测：超出物理合理范围 [2.0, 6.0] V
        if not np.isnan(v_pred) and (v_pred < 2.0 or v_pred > 6.0):
            if not voltage_anomaly:
                print(f"    ⚠ 电压异常: step={step + 1}, pred={v_pred:.2f}V, set={v_set:.2f}V")
                voltage_anomaly = True
            done = True  # 异常时提前终止

        history['voltage_pred'].append(v_pred)
        history['voltage_set'].append(v_set)
        history['voltage_actual'].append(
            sample['actual_voltage'][step] if step < len(sample['actual_voltage']) else np.nan)
        history['alf_action'].append(info['day1_action'][0])
        history['out_action'].append(info['day1_action'][1])
        history['voltage_error'].append(
            abs(v_pred - v_set) if not (np.isnan(v_pred) or v_set is None or np.isnan(v_set)) else np.nan)

        state = next_state
        if done:
            break

    return history


def plot_optimization_process(history, sample, save_path):
    """主图表：4面板综合展示"动作调整 → 电压响应 → 逼近设定值"的优化全过程"""
    n_steps = len(history['voltage_pred'])
    days = np.arange(1, n_steps + 1)
    pred = np.array(history['voltage_pred'])
    set_v = np.array(history['voltage_set'])
    alf = np.array(history['alf_action'])
    out = np.array(history['out_action'])

    valid_mask = ~(np.isnan(pred) | np.isnan(set_v))
    mae = np.mean(np.abs(pred[valid_mask] - set_v[valid_mask])) if valid_mask.any() else 0

    start_date = pd.Timestamp(sample['dates_future'][0]).strftime('%Y-%m-%d')
    pot_num = sample['pot_num']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, hspace=0.12, height_ratios=[2.5, 1.2, 1.5, 1.5])

    # ===== 面板1: 电压追踪 — 核心优化效果 =====
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(days, pred, 'o-', color=COLOR_VOLTAGE, linewidth=2.5, markersize=9,
             label='PPO优化后工作平均电压', zorder=3)
    ax1.plot(days, set_v, 's--', color=COLOR_SETPOINT, linewidth=2.2, markersize=8,
             label='实际设定电压（基准线）', zorder=2)

    # 误差填充
    if valid_mask.any():
        ax1.fill_between(days, pred, set_v, alpha=0.12, color=COLOR_ERROR_FILL,
                         label='电压偏差区域')
    # 收敛箭头标注
    if n_steps >= 7:
        mid = n_steps // 2
        early_err = np.mean(np.abs(pred[:3] - set_v[:3])) if np.sum(valid_mask[:3]) > 0 else 0
        late_err = np.mean(np.abs(pred[-3:] - set_v[-3:])) if np.sum(valid_mask[-3:]) > 0 else 0
        if late_err < early_err and n_steps > 6:
            ax1.annotate('', xy=(n_steps - 1, pred[-1]), xytext=(2, pred[0]),
                         arrowprops=dict(arrowstyle='->', color=COLOR_CONVERGE,
                                         lw=2.5, connectionstyle='arc3,rad=.2'),
                         fontsize=11)
            ax1.text(n_steps / 2 + 1, ax1.get_ylim()[0] +
                     (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.92,
                     '← 向设定电压收敛', fontsize=11, color=COLOR_CONVERGE, fontweight='bold',
                     ha='center')

    # MAE标注框
    ax1.text(0.02, 0.95, f'MAE = {mae:.4f} V | 步数 = {n_steps}',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='gray', alpha=0.9))

    ax1.set_ylabel('电压 (V)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_title(f'槽号 {pot_num} — 起始日期: {start_date} — 电压优化效果',
                  fontsize=14, fontweight='bold', pad=10)

    # ===== 面板2: 电压逐日偏差 =====
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    errors = np.array(history['voltage_error'])
    colors_bar = ['#27AE60' if e < 0.02 else '#F39C12' if e < 0.05 else '#E74C3C'
                  for e in errors]
    bars = ax2.bar(days, errors * 1000, color=colors_bar, alpha=0.85, edgecolor='white',
                   linewidth=0.5)  # 转换为mV
    ax2.axhline(y=20, color='#27AE60', linestyle='--', linewidth=1.2, alpha=0.7,
                label='20 mV (优)')
    ax2.axhline(y=50, color='#F39C12', linestyle='--', linewidth=1.2, alpha=0.7,
                label='50 mV (可接受)')
    ax2.set_ylabel('电压偏差\n(mV)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')

    # ===== 面板3: ALF加料量调整 =====
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _plot_action_panel(ax3, days, alf, 'ALF加料量(实际)', 'kg',
                       COLOR_ALF, ACTION_ALF_MAX_CHANGE, alf_max_change=ACTION_ALF_MAX_CHANGE)

    # ===== 面板4: 实际出铝量调整 =====
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _plot_action_panel(ax4, days, out, '实际出铝量', 'kg',
                       COLOR_OUT, ACTION_OUT_MAX_CHANGE, alf_max_change=None)

    # X轴
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0.5, max(n_steps, 14) + 0.5)
    ax4.set_xlabel('优化步数 (天)', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(1, max(n_steps, 14) + 1))

    # 隐藏面板1/2的x轴标签
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(labelbottom=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图表已保存: {save_path}  (MAE={mae:.4f}V, {n_steps}步)")


def _plot_action_panel(ax, days, values, label, unit, color, max_change, alf_max_change=None):
    """绘制单个动作变量的调整趋势面板"""
    mean_val = np.nanmean(values)
    ax.plot(days, values, 'o-', color=color, linewidth=2.2, markersize=8,
            label=f'{label}', zorder=3)
    ax.axhline(y=mean_val, color=COLOR_MEAN, linestyle=':', linewidth=1.5, alpha=0.8,
               label=f'均值 {mean_val:.1f} {unit}')
    ax.fill_between(days, mean_val - max_change, mean_val + max_change,
                    alpha=0.06, color=color, label=f'±{max_change}{unit} 变化容忍带')

    # 标注变化方向
    for i in range(1, len(days)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue
        diff = values[i] - values[i - 1]
        if abs(diff) > (max_change * 0.3 if alf_max_change is None else alf_max_change * 0.3):
            arrow_color = COLOR_INCREASE if diff > 0 else COLOR_DECREASE
            arrow = '▲' if diff > 0 else '▼'
            ax.annotate(arrow, (days[i], values[i]),
                        textcoords="offset points", xytext=(0, 12 if diff > 0 else -18),
                        ha='center', fontsize=9, color=arrow_color, fontweight='bold')

    ax.set_ylabel(f'{label}\n({unit})', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')


def plot_overview(all_results, save_path):
    """摘要面板：展示所有样本的MAE与关键指标"""
    n = len(all_results)
    if n == 0:
        return

    mae_list = []
    labels = []
    for history, sample in all_results:
        pred = np.array(history['voltage_pred'])
        set_v = np.array(history['voltage_set'])
        valid = ~(np.isnan(pred) | np.isnan(set_v))
        mae = np.mean(np.abs(pred[valid] - set_v[valid])) if valid.any() else 0
        mae_list.append(mae * 1000)
        labels.append(f"{sample['pot_num']}\n{str(sample['dates_future'][0])[:10]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 子图1: 各样本MAE(mV)排名
    ax = axes[0]
    sorted_idx = np.argsort(mae_list)
    colors = ['#27AE60' if m < 20 else '#F39C12' if m < 50 else '#E74C3C' for m in
              np.array(mae_list)[sorted_idx]]
    ax.barh(range(n), np.array(mae_list)[sorted_idx], color=colors, alpha=0.85,
            edgecolor='white')
    ax.set_yticks(range(n))
    ax.set_yticklabels([labels[i] for i in sorted_idx], fontsize=8)
    ax.axvline(x=20, color='#27AE60', linestyle='--', linewidth=1.2, label='20 mV (优)')
    ax.axvline(x=50, color='#F39C12', linestyle='--', linewidth=1.2, label='50 mV')
    ax.set_xlabel('MAE (mV)', fontsize=12, fontweight='bold')
    ax.set_title('各样本 MAE 排名 (vs 设定电压)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')

    # 子图2: 各测试槽平均MAE
    ax = axes[1]
    pot_mae = {}
    for i, (h, s) in enumerate(all_results):
        pot = s['pot_num']
        pot_mae.setdefault(pot, []).append(mae_list[i])
    pot_nums = sorted(pot_mae.keys())
    pot_avg = [np.mean(pot_mae[p]) for p in pot_nums]
    pot_std = [np.std(pot_mae[p]) for p in pot_nums]
    bars = ax.bar(range(len(pot_nums)), pot_avg, yerr=pot_std, capsize=5,
                  color='steelblue', alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(pot_nums)))
    ax.set_xticklabels(pot_nums, fontsize=10)
    ax.set_ylabel('MAE (mV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('槽号', fontsize=12, fontweight='bold')
    ax.set_title('各测试槽平均 MAE (含±1σ)', fontsize=13, fontweight='bold')
    ax.axhline(y=20, color='#27AE60', linestyle='--', linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    # 子图3: 分布直方图
    ax = axes[2]
    ax.hist(mae_list, bins=min(15, n), color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(x=20, color='#27AE60', linestyle='--', linewidth=1.5, label='20 mV')
    ax.axvline(x=50, color='#E74C3C', linestyle='--', linewidth=1.5, label='50 mV')
    ax.set_xlabel('MAE (mV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('样本数', fontsize=12, fontweight='bold')
    ax.set_title(f'MAE分布 (n={n}, avg={np.mean(mae_list):.1f} mV)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    fig.suptitle('PPO电压优化 — 测试集评估总览', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"总览图已保存: {save_path}")


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("=" * 60)
    print("PPO优化结果可视化")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    test_pots = TEST_POTS
    print(f"   测试槽号: {test_pots}")
    samples, scaler, feature_cols = load_test_data(DATA_PATH, test_pots)
    print(f"   测试样本总数: {len(samples)}")

    # 2. 加载条件预测模型
    print("\n2. 加载条件预测模型...")
    num_features = len(feature_cols)
    num_pots = len(TRAIN_POTS) + len(VAL_POTS) + len(TEST_POTS)
    predictor = load_predictor_model(PREDICTOR_MODEL_PATH, num_pots, num_features, device)
    print("   条件预测模型加载完成")

    # 3. 创建环境
    print("\n3. 创建仿真环境...")
    env = VoltageControlEnv(predictor, scaler, feature_cols, device)
    print(f"   输入特征维度: {len(feature_cols)}")

    # 4. 加载PPO模型
    print("\n4. 加载PPO模型...")
    ppo_agent = MPDPPO(num_features, num_pots, ACTION_TRAJECTORY_DIM, device)
    ppo_model_path = os.path.join(OUTPUT_DIR, 'best_ppo_model.pth')
    if not os.path.exists(ppo_model_path):
        # 尝试final模型
        ppo_model_path = os.path.join(OUTPUT_DIR, 'final_ppo_model.pth')
        if not os.path.exists(ppo_model_path):
            print(f"   错误: 未找到PPO模型文件 ({OUTPUT_DIR})")
            return
    ppo_agent.load_model(ppo_model_path)
    ppo_agent.actor.eval()
    print(f"   PPO模型加载完成: {os.path.basename(ppo_model_path)}")

    # 5. 选择测试样本（每个槽取最近2个，共12个）
    print("\n5. 运行评估并生成图表...")
    test_pot_nums = sorted(set(s['pot_num'] for s in samples))
    samples_per_pot = 2
    selected = []
    for pot in test_pot_nums:
        pot_samples = [s for s in samples if s['pot_num'] == pot]
        selected.extend(pot_samples[-samples_per_pot:])
    print(f"   选定评估样本数: {len(selected)}")

    output_dir = os.path.join(OUTPUT_DIR, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for i, sample in enumerate(selected):
        pot_num = sample['pot_num']
        start_date = pd.Timestamp(sample['dates_future'][0]).strftime('%Y%m%d')
        print(f"\n  评估 [{i + 1}/{len(selected)}] 槽{pot_num} 起始{start_date}...")

        history = run_evaluation_episode(env, ppo_agent, sample)

        # 诊断输出（配对过滤，避免数组长度不匹配）
        pred_arr = np.array(history['voltage_pred'])
        set_arr = np.array(history['voltage_set'])
        valid = ~(np.isnan(pred_arr) | np.isnan(set_arr))
        n_valid = valid.sum()
        if n_valid > 0:
            mae = np.mean(np.abs(pred_arr[valid] - set_arr[valid]))
            print(f"    有效步数: {n_valid}, MAE vs 设定: {mae:.4f} V ({mae * 1000:.1f} mV)")
            print(f"    电压范围: pred={pred_arr[valid].min():.2f}~{pred_arr[valid].max():.2f}V, "
                  f"set={set_arr[valid].min():.2f}~{set_arr[valid].max():.2f}V")
            print(f"    ALF: {np.nanmin(history['alf_action']):.1f}~{np.nanmax(history['alf_action']):.1f} "
                  f"kg, 出铝量: {np.nanmin(history['out_action']):.0f}~{np.nanmax(history['out_action']):.0f} kg")

        save_path = os.path.join(output_dir, f'pot{pot_num}_{start_date}_rollout.png')
        plot_optimization_process(history, sample, save_path)
        all_results.append((history, sample))

    # 6. 生成总览图
    print("\n6. 生成总览图...")
    overview_path = os.path.join(output_dir, 'summary_overview.png')
    plot_overview(all_results, overview_path)

    # 7. 打印汇总
    print("\n" + "=" * 60)
    print("可视化完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    mae_all = []
    for h, s in all_results:
        p = np.array(h['voltage_pred'])
        sv = np.array(h['voltage_set'])
        v = ~(np.isnan(p) | np.isnan(sv))
        if v.any():
            mae_all.append(np.mean(np.abs(p[v] - sv[v])) * 1000)
    if mae_all:
        print(f"\n样本数: {len(all_results)}")
        print(f"MAE (mV) — 均值: {np.mean(mae_all):.1f}, "
              f"中位数: {np.median(mae_all):.1f}, "
              f"最小: {np.min(mae_all):.1f}, "
              f"最大: {np.max(mae_all):.1f}")


if __name__ == '__main__':
    main()
