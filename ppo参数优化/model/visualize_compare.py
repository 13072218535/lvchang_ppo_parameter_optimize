"""
多算法对比可视化脚本
保持现有4面板排版，叠加不同算法的优化曲线
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from config import *
from environment import VoltageControlEnv, load_predictor_model
from algorithms import create_algorithm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 6种算法配色
ALGO_COLORS = {
    'mpd_ppo': '#1A73E8',
    'vanilla_ppo': '#DC3912',
    'a2c': '#888888',
    'ddpg': '#109618',
    'td3': '#FF9900',
    'sac': '#990099',
    'taa_ppo': '#00BCD4',
    'taa_ppo_4d': '#E91E63',
}
ALGO_NAMES = {
    'mpd_ppo': 'MPD-PPO',
    'vanilla_ppo': 'Vanilla PPO',
    'a2c': 'A2C',
    'ddpg': 'DDPG',
    'td3': 'TD3',
    'sac': 'SAC',
    'taa_ppo': 'TAA-PPO',
    'taa_ppo_4d': 'TAA-PPO-4D',
}
ALGO_LINESTYLES = {
    'mpd_ppo': '-',
    'vanilla_ppo': '--',
    'a2c': ':',
    'ddpg': '-.',
    'td3': (0, (3, 1, 1, 1)),
    'sac': (0, (5, 2)),
    'taa_ppo': '-',
    'taa_ppo_4d': '-.',
}

SCALER_PATH = os.path.join(PROJECT_ROOT, 'model', 'output', 'scaler.pkl')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'model', 'output')


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


def load_test_data(data_path, test_pots):
    df = pd.read_excel(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    df, feature_cols = prepare_raw_features(df)
    all_pots = sorted(df['槽号'].unique())
    pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

    samples = []
    for pot_id in test_pots:
        if pot_id not in df['槽号'].unique():
            continue
        pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
        if len(pot_data) < INPUT_LEN + OUTPUT_LEN:
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
                'pot_id': pot_to_idx[pot_id], 'pot_num': pot_id,
                'dates_past': pot_dates[i:i + INPUT_LEN],
                'dates_future': pot_dates[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
            })
    return samples, scaler, feature_cols


def run_evaluation_episode(env, algo, sample, algo_name):
    """运行评估episode，返回history dict"""
    state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
    history = {
        'voltage_pred': [], 'voltage_set': [], 'alf_action': [], 'out_action': [],
        'voltage_error': [], 'voltage_actual': [],
    }
    is_off_policy = algo_name in ['ddpg', 'td3', 'sac']

    for step in range(MAX_EPISODE_STEPS):
        result = algo.select_action(state)
        if is_off_policy:
            action = result  # 只返回action数组
        else:
            action = result[0]
        next_state, reward, done, info = env.step(action)
        v_pred = info['voltage_pred']
        v_set = info['target_voltage_set'] if info['target_voltage_set'] is not None else np.nan

        if not np.isnan(v_pred) and (v_pred < 2.0 or v_pred > 6.0):
            done = True

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


def plot_compare_sample(all_histories, sample, save_path, algos):
    """4面板叠加图：同一测试样本，各算法叠加展示"""
    n_steps = max(len(h['voltage_pred']) for h in all_histories.values() if h is not None)
    days = np.arange(1, n_steps + 1)
    start_date = pd.Timestamp(sample['dates_future'][0]).strftime('%Y-%m-%d')
    pot_num = sample['pot_num']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, hspace=0.12, height_ratios=[2.5, 1.2, 1.5, 1.5])

    # 面板1: 电压追踪叠加
    ax1 = fig.add_subplot(gs[0])
    set_v = None
    for algo_name in algos:
        h = all_histories.get(algo_name)
        if h is None: continue
        pred = np.array(h['voltage_pred'])
        if set_v is None:
            set_v = np.array(h['voltage_set'])
        n = len(pred)
        ax1.plot(np.arange(1, n + 1), pred, linestyle=ALGO_LINESTYLES[algo_name],
                 color=ALGO_COLORS[algo_name], linewidth=2.0, markersize=6,
                 marker='o', label=ALGO_NAMES[algo_name], alpha=0.85)
    if set_v is not None:
        ax1.plot(np.arange(1, len(set_v) + 1), set_v, 's--', color='black',
                 linewidth=2.5, markersize=8, label='Set Voltage (Target)', zorder=10)

    # 各算法MAE标注
    mae_texts = []
    for algo_name in algos:
        h = all_histories.get(algo_name)
        if h is None: continue
        pred = np.array(h['voltage_pred']); sv = np.array(h['voltage_set'])
        valid = ~(np.isnan(pred) | np.isnan(sv))
        if valid.any():
            mae = np.mean(np.abs(pred[valid] - sv[valid])) * 1000
            mae_texts.append(f'{ALGO_NAMES[algo_name]}: {mae:.0f} mV')

    ax1.text(0.02, 0.95, ' | '.join(mae_texts), transform=ax1.transAxes,
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    ax1.set_ylabel('Voltage (V)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_title(f'Pot {pot_num} — {start_date} — Multi-Algorithm Comparison',
                  fontsize=14, fontweight='bold', pad=10)

    # 面板2: 电压偏差（分组柱状图）
    ax2 = fig.add_subplot(gs[1])
    n_algos = len([a for a in algos if all_histories.get(a) is not None])
    bar_width = 0.8 / n_algos
    for j, algo_name in enumerate(algos):
        h = all_histories.get(algo_name)
        if h is None: continue
        errors = np.array(h['voltage_error']) * 1000  # mV
        n_e = len(errors)
        x_pos = np.arange(1, n_e + 1) + (j - (n_algos - 1) / 2) * bar_width
        ax2.bar(x_pos, errors, bar_width * 0.9, color=ALGO_COLORS[algo_name],
                alpha=0.85, label=ALGO_NAMES[algo_name], edgecolor='white', linewidth=0.3)

    ax2.axhline(y=20, color='#27AE60', linestyle='--', linewidth=1.2, alpha=0.7, label='20 mV (Good)')
    ax2.axhline(y=50, color='#F39C12', linestyle='--', linewidth=1.2, alpha=0.7, label='50 mV (OK)')
    ax2.set_ylabel('Error (mV)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')

    # 面板3: ALF叠加
    ax3 = fig.add_subplot(gs[2])
    for algo_name in algos:
        h = all_histories.get(algo_name)
        if h is None: continue
        alf = np.array(h['alf_action'])
        ax3.plot(np.arange(1, len(alf) + 1), alf, linestyle=ALGO_LINESTYLES[algo_name],
                 color=ALGO_COLORS[algo_name], linewidth=2.0, marker='o', markersize=5,
                 label=ALGO_NAMES[algo_name], alpha=0.85)
    ax3.set_ylabel('ALF (kg)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.25, linestyle='--')

    # 面板4: OUT叠加
    ax4 = fig.add_subplot(gs[3])
    for algo_name in algos:
        h = all_histories.get(algo_name)
        if h is None: continue
        out = np.array(h['out_action'])
        ax4.plot(np.arange(1, len(out) + 1), out, linestyle=ALGO_LINESTYLES[algo_name],
                 color=ALGO_COLORS[algo_name], linewidth=2.0, marker='o', markersize=5,
                 label=ALGO_NAMES[algo_name], alpha=0.85)
    ax4.set_xlabel('Step (day)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Output (kg)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, ncol=3)
    ax4.grid(True, alpha=0.25, linestyle='--')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0.5, max(n_steps, 14) + 0.5)
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(labelbottom=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {save_path}")


def plot_overview_compare(all_mae_data, save_path, algos, human_baseline=0):
    """总览图：各算法MAE分布对比（箱线图 + 排名） + 人类基线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 子图1: MAE排名（含人类基线）
    ax = axes[0]
    algo_avg_mae = {}
    for algo_name in algos:
        if algo_name in all_mae_data:
            mae_list = [m for m in all_mae_data[algo_name] if not np.isnan(m)]
            if mae_list:
                algo_avg_mae[algo_name] = np.mean(mae_list)
            else:
                algo_avg_mae[algo_name] = float('inf')

    sorted_algos = sorted(algo_avg_mae.keys(), key=lambda a: algo_avg_mae[a])
    x_labels = [ALGO_NAMES[a] for a in sorted_algos] + ['Human Baseline']
    y_values = [algo_avg_mae[a] for a in sorted_algos] + [human_baseline]
    colors = [ALGO_COLORS[a] for a in sorted_algos] + ['#333333']
    alphas = [0.85] * len(sorted_algos) + [1.0]
    bar_heights = [0.6] * len(sorted_algos) + [0.4]
    for j in range(len(y_values)):
        ax.barh(j, y_values[j], height=bar_heights[j], color=colors[j], alpha=alphas[j], edgecolor='white')
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(x_labels, fontsize=11)
    ax.set_xlabel('MAE (mV)', fontsize=12, fontweight='bold')
    ax.set_title('MAE Ranking vs Human Operator', fontsize=13, fontweight='bold')
    ax.axvline(x=20, color='#27AE60', linestyle='--', linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')

    # 子图2: MAE箱线图
    ax = axes[1]
    box_data = []
    box_labels = []
    for algo_name in algos:
        if algo_name in all_mae_data:
            mae_list = [m for m in all_mae_data[algo_name] if not np.isnan(m)]
            if mae_list:
                box_data.append(mae_list)
                box_labels.append(ALGO_NAMES[algo_name])
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showfliers=True)
    for patch, algo_name in zip(bp['boxes'], [a for a in algos if a in all_mae_data and all_mae_data[a]]):
        patch.set_facecolor(ALGO_COLORS[algo_name])
        patch.set_alpha(0.6)
    ax.set_ylabel('MAE (mV)', fontsize=12, fontweight='bold')
    ax.set_title('MAE Distribution by Algorithm', fontsize=13, fontweight='bold')
    ax.axhline(y=20, color='#27AE60', linestyle='--', linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_fontsize(9)

    # 子图3: 汇总表格（含人类基线对比）
    ax = axes[2]
    ax.axis('off')
    table_data = []
    for algo_name in algos:
        if algo_name in all_mae_data:
            mae_list = [m for m in all_mae_data[algo_name] if not np.isnan(m)]
            if mae_list:
                avg = np.mean(mae_list)
                impr = ((human_baseline - avg) / human_baseline * 100) if human_baseline > 0 else 0
                table_data.append([
                    ALGO_NAMES[algo_name],
                    f'{avg:.1f}',
                    f'{np.std(mae_list):.1f}',
                    f'{np.min(mae_list):.1f}',
                    f'{np.max(mae_list):.1f}',
                    f'{impr:+.1f}%',
                ])
    if human_baseline > 0:
        table_data.append(['Human (Baseline)', f'{human_baseline:.1f}', '-', '-', '-', '---'])
    if table_data:
        table = ax.table(cellText=table_data,
                         colLabels=['Algorithm', 'Mean', 'Std', 'Min', 'Max', 'vs Human'],
                         cellLoc='center', loc='center',
                         colWidths=[0.18, 0.11, 0.11, 0.11, 0.11, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('#CCCCCC')
            if cell.get_text().get_text().startswith('+'):
                cell.get_text().set_color('#27AE60')
                cell.get_text().set_fontweight('bold')
    ax.set_title('MAE Summary vs Human Operator', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle('Multi-Algorithm Comparison — Overview', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Overview saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Algorithm Comparison Visualization')
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo', 'taa_ppo_4d', 'ddpg', 'td3', 'sac'],
                        help='Algorithms to compare (must have trained models)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Samples per test pot (default: 5, max: all available)')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # 检查可用算法（有训练好的模型）
    avail_algos = []
    for algo_name in args.algos:
        model_file = os.path.join(OUTPUT_DIR, algo_name, 'best_model.pth')
        if not os.path.exists(model_file):
            model_file = os.path.join(OUTPUT_DIR, algo_name, 'final_model.pth')
        if os.path.exists(model_file):
            avail_algos.append(algo_name)
        else:
            print(f"Skip {algo_name}: no model found at {os.path.join(OUTPUT_DIR, algo_name)}")

    if len(avail_algos) < 1:
        print("Error: No trained models found. Train at least one algorithm first:")
        print("  python train_compare.py --algo mpd_ppo --epochs 200")
        return

    print(f"Algorithms to compare: {[ALGO_NAMES[a] for a in avail_algos]}")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n1. Loading test data...")
    test_pots = TEST_POTS
    samples, scaler, feature_cols = load_test_data(DATA_PATH, test_pots)
    print(f"   Test samples: {len(samples)}")

    # 2. 加载条件预测器（自动检测ensemble）
    print("\n2. Loading predictor...")
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
        print(f"   Found {len(ensemble_paths)} ensemble checkpoints (< {UNC_NUM_ENSEMBLE}), trying single predictor")
        model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_conditional_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'final_conditional_model.pth')
        print(f"   Model: {model_path}")
        predictor = load_predictor_model(model_path, num_pots, num_features, device)
    print("   Predictor loaded.")

    # 3. 加载所有算法模型
    print("\n3. Loading algorithm models...")
    env = VoltageControlEnv(predictor, scaler, feature_cols, device,
                            ensemble_is_ensemble=ensemble_is_ensemble)
    models = {}
    for algo_name in avail_algos:
        model_file = os.path.join(OUTPUT_DIR, algo_name, 'best_model.pth')
        if not os.path.exists(model_file):
            model_file = os.path.join(OUTPUT_DIR, algo_name, 'final_model.pth')
        algo = create_algorithm(algo_name, num_features, num_pots, ACTION_TRAJECTORY_DIM, device)
        algo.load_model(model_file)
        if hasattr(algo, 'actor'):
            algo.actor.eval()
        print(f"   {ALGO_NAMES[algo_name]}: {os.path.basename(model_file)}")
        models[algo_name] = algo

    # 4. 运行对比评估
    print("\n4. Running comparison evaluation...")
    output_dir = os.path.join(OUTPUT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    # 每个测试槽取最近N个样本
    test_pot_nums = sorted(set(s['pot_num'] for s in samples))
    n_per_pot = min(args.samples, min(
        sum(1 for s in samples if s['pot_num'] == p) for p in test_pot_nums))
    selected = []
    for pot in test_pot_nums:
        pot_samples = [s for s in samples if s['pot_num'] == pot]
        # 时间上均匀采样，覆盖不同时间段
        step = max(1, len(pot_samples) // n_per_pot)
        indices = list(range(len(pot_samples) - 1, -1, -step))[:n_per_pot]
        selected.extend([pot_samples[i] for i in sorted(indices)])
    print(f"   Selected samples: {len(selected)} ({n_per_pot} per pot × {len(test_pot_nums)} pots)")

    all_mae_data = {algo_name: [] for algo_name in avail_algos}
    all_human_mae = []  # human baseline MAE per sample

    for i, sample in enumerate(selected):
        pot_num = sample['pot_num']
        start_date = pd.Timestamp(sample['dates_future'][0]).strftime('%Y%m%d')
        print(f"\n  [{i + 1}/{len(selected)}] Pot {pot_num} {start_date}")

        # Human baseline: MAE of actual historical voltage vs set voltage
        av = np.array(sample['actual_voltage'][:MAX_EPISODE_STEPS])
        sv_h = np.array(sample['target_voltage'][:MAX_EPISODE_STEPS])
        valid_h = ~(np.isnan(av) | np.isnan(sv_h))
        human_mae_sample = np.mean(np.abs(av[valid_h] - sv_h[valid_h])) * 1000 if valid_h.any() else np.nan
        all_human_mae.append(human_mae_sample)
        human_str = f"Human: MAE={human_mae_sample:.1f} mV" if not np.isnan(human_mae_sample) else "Human: N/A"
        print(f"     {human_str}")

        all_histories = {}
        for algo_name in avail_algos:
            algo = models[algo_name]
            env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
            history = run_evaluation_episode(env, algo, sample, algo_name)

            pred = np.array(history['voltage_pred'])
            sv = np.array(history['voltage_set'])
            valid = ~(np.isnan(pred) | np.isnan(sv))
            if valid.any():
                mae = np.mean(np.abs(pred[valid] - sv[valid])) * 1000
                all_mae_data[algo_name].append(mae)
                impr = ((human_mae_sample - mae) / human_mae_sample * 100) if (not np.isnan(human_mae_sample) and human_mae_sample > 0) else 0
                print(f"     {ALGO_NAMES[algo_name]}: MAE={mae:.1f} mV  (vs Human: {impr:+.1f}%)")
            all_histories[algo_name] = history

        save_path = os.path.join(output_dir, f'pot{pot_num}_{start_date}_compare.png')
        plot_compare_sample(all_histories, sample, save_path, avail_algos)

    # Compute human baseline stats
    human_valid = [m for m in all_human_mae if not np.isnan(m)]
    human_mean = np.mean(human_valid) if human_valid else 0
    print(f"\n  Human Baseline: Mean={human_mean:.1f} mV (n={len(human_valid)} samples)")

    # 5. 生成总览对比图
    print("\n5. Generating overview comparison...")
    overview_path = os.path.join(output_dir, 'comparison_overview.png')
    plot_overview_compare(all_mae_data, overview_path, avail_algos, human_mean)

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Output: {output_dir}")
    print(f"  Human Baseline: {human_mean:.1f} mV")
    for algo_name in avail_algos:
        mae_list = [m for m in all_mae_data[algo_name] if not np.isnan(m)]
        if mae_list:
            avg = np.mean(mae_list)
            impr = ((human_mean - avg) / human_mean * 100) if human_mean > 0 else 0
            win_rate = np.mean([1 if m < h else 0 for m, h in zip(mae_list, human_valid)]) * 100
            print(f"  {ALGO_NAMES[algo_name]}: Mean={avg:.1f} mV  "
                  f"vs Human: {impr:+.1f}%  Win Rate: {win_rate:.0f}%")


if __name__ == '__main__':
    main()
