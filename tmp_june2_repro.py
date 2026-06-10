"""Reproduce June 2 experiment: weak-predictor-trained policies vs strong predictors"""
import os, sys
for k in list(sys.modules.keys()):
    if 'config' in k.lower(): del sys.modules[k]

PROJECT = r'E:\ClaudeCodeWorkplace\2026-5-12-参数优化'
RL_DIR = os.path.join(PROJECT, 'ppo参数优化', 'model')
MODEL_OUT = os.path.join(PROJECT, 'model', 'output')
os.chdir(RL_DIR)
sys.path.insert(0, RL_DIR)
sys.path.insert(1, os.path.join(PROJECT, 'model'))

import config as rl_cfg
import torch, pickle, numpy as np, pandas as pd, glob
from environment import VoltageControlEnv, load_predictor_model
from algorithms import create_algorithm

device = torch.device('cuda')
cols = rl_cfg.HIGH_CORR_FEATURES.copy()
nf, npots = len(cols), 42
with open(os.path.join(MODEL_OUT, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)

# Build test samples (same as visualize_compare)
df = pd.read_excel(rl_cfg.DATA_PATH); df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['槽号', '日期']).reset_index(drop=True)
for col in cols:
    if col in df.columns:
        df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
        df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
        df[col] = df[col].fillna(df[col].mean())
all_pots = sorted(df['槽号'].unique())
pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

samples = []
for pot_id in rl_cfg.TEST_POTS:
    if pot_id not in df['槽号'].unique(): continue
    pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
    if len(pot_data) < rl_cfg.INPUT_LEN + rl_cfg.OUTPUT_LEN: continue
    pf = pot_data[cols].values
    sv = pot_data['实际设定'].values if '实际设定' in pot_data.columns else pot_data[rl_cfg.TARGET].values
    for i in range(len(pot_data) - rl_cfg.INPUT_LEN - rl_cfg.OUTPUT_LEN + 1):
        samples.append({'past_features': pf[i:i+7], 'target_voltage': sv[i+7:i+21],
                        'pot_id': pot_to_idx[pot_id], 'pot_num': pot_id})

selected = []
for pot in rl_cfg.TEST_POTS:
    pot_samples = [s for s in samples if s['pot_num'] == pot]
    step = max(1, len(pot_samples) // 3)
    indices = list(range(len(pot_samples) - 1, -1, -step))[:3]
    selected.extend([pot_samples[i] for i in sorted(indices)])
print(f'Test: {len(selected)} samples')

ALGOS = ['mpd_ppo', 'vanilla_ppo', 'taa_ppo', 'taa_ppo_4d', 'a2c', 'ddpg', 'td3', 'sac']
NAMES = {'mpd_ppo':'MPD-PPO','vanilla_ppo':'Vanilla PPO','a2c':'A2C','ddpg':'DDPG',
         'td3':'TD3','sac':'SAC','taa_ppo':'TAA-PPO','taa_ppo_4d':'TAA-PPO-4D'}
POLICY_DIR = os.path.join(RL_DIR, 'output', 'single_model_policies')

# Predictors
pred_single = load_predictor_model(
    os.path.join(MODEL_OUT, 'ensemble_baseline_no_aug', 'best_conditional_seed2.pth'),
    npots, nf, device)
paths_noaug = sorted(glob.glob(os.path.join(MODEL_OUT, 'ensemble_baseline_no_aug', 'best_conditional_seed*.pth')))
pred_noaug = load_predictor_model(None, npots, nf, device, ensemble=True, ensemble_checkpoint_paths=paths_noaug)
paths_aug = sorted(glob.glob(os.path.join(MODEL_OUT, 'best_conditional_seed*.pth')))
pred_aug = load_predictor_model(None, npots, nf, device, ensemble=True, ensemble_checkpoint_paths=paths_aug)

def eval_policy(algo_name, predictor, is_ensemble):
    policy_path = os.path.join(POLICY_DIR, f'{algo_name}_best.pth')
    algo = create_algorithm(algo_name, nf, npots, rl_cfg.ACTION_TRAJECTORY_DIM, device)
    algo.load_model(policy_path)
    if hasattr(algo, 'actor'): algo.actor.eval()
    env = VoltageControlEnv(predictor, scaler, cols, device, ensemble_is_ensemble=is_ensemble)
    is_off = algo_name in ['ddpg', 'td3', 'sac']
    errors = []
    for s in selected:
        state = env.reset(s['past_features'], s['target_voltage'], s['pot_id'])
        for step in range(rl_cfg.MAX_EPISODE_STEPS):
            res = algo.select_action(state)
            action = res if is_off else res[0]
            ns, reward, done, info = env.step(action)
            vp = info['voltage_pred']; vs = info['target_voltage_set']
            if vs is not None and not np.isnan(vp) and not np.isnan(vs):
                errors.append(abs(vp - vs))
            state = ns
            if done: break
    return np.mean(errors) * 1000

print()
print('=' * 90)
print('JUNE 2 REPRODUCTION: Weak-predictor-trained policies vs Strong predictors')
print('=' * 90)

results = {}
for algo_name in ALGOS:
    m_single = eval_policy(algo_name, pred_single, False)
    m_noaug  = eval_policy(algo_name, pred_noaug, True)
    m_aug    = eval_policy(algo_name, pred_aug, True)
    results[algo_name] = (m_single, m_noaug, m_aug)

print()
print(f'{"Algorithm":<18} {"Single":>8} {"NonAug Ens":>11} {"Aug Ens":>8} {"vs Single":>10} {"vs NonAug":>10}')
print('-' * 70)
sums = [0, 0, 0]
for algo_name in ALGOS:
    m1, m2, m3 = results[algo_name]
    sums[0] += m1; sums[1] += m2; sums[2] += m3
    print(f'{NAMES[algo_name]:<18} {m1:7.1f}mV {m2:10.1f}mV {m3:7.1f}mV {(1-m3/m1)*100:+9.1f}% {(1-m3/m2)*100:+9.1f}%')
avgs = [s/8 for s in sums]
print('-' * 70)
mean_label = 'MEAN'
print(f'{mean_label:<18} {avgs[0]:7.1f}mV {avgs[1]:10.1f}mV {avgs[2]:7.1f}mV {(1-avgs[2]/avgs[0])*100:+9.1f}% {(1-avgs[2]/avgs[1])*100:+9.1f}%')

print()
print('Key: Single = weak single-model predictor trained policies')
print('      NonAug Ens = same policies evaluated with non-augmented ensemble')
print('      Aug Ens = same policies evaluated with augmented (fixed) ensemble')
print('      vs Single = improvement over training predictor (June 2 equivalent metric)')
