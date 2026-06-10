"""Fast MAE comparison: all 8 RL policies × 2 predictors (no chart generation)"""
import os, sys, time

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
from ppo import MPDPPO
from algorithms import create_algorithm

device = torch.device('cuda')
cols = rl_cfg.HIGH_CORR_FEATURES.copy()
nf = len(cols)
npots = 42

# Load scaler
with open(os.path.join(MODEL_OUT, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Build test samples
df = pd.read_excel(rl_cfg.DATA_PATH)
df['日期'] = pd.to_datetime(df['日期'])
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

# 3 per pot — SAME sampling as visualize_compare.py (uniform across time)
selected = []
n_per_pot = 3
for pot in rl_cfg.TEST_POTS:
    pot_samples = [s for s in samples if s['pot_num'] == pot]
    step = max(1, len(pot_samples) // n_per_pot)
    indices = list(range(len(pot_samples) - 1, -1, -step))[:n_per_pot]
    selected.extend([pot_samples[i] for i in sorted(indices)])
print(f'Test: {len(selected)} samples ({len(rl_cfg.TEST_POTS)} pots × ~3)\n')

ALGOS = ['mpd_ppo', 'vanilla_ppo', 'taa_ppo', 'taa_ppo_4d', 'a2c', 'ddpg', 'td3', 'sac']
ALGO_NAMES = {'mpd_ppo':'MPD-PPO','vanilla_ppo':'Vanilla PPO','a2c':'A2C','ddpg':'DDPG',
              'td3':'TD3','sac':'SAC','taa_ppo':'TAA-PPO','taa_ppo_4d':'TAA-PPO-4D'}

def load_predictor(ckpt_dir):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'best_conditional_seed*.pth')))
    return load_predictor_model(None, npots, nf, device, ensemble=True, ensemble_checkpoint_paths=paths)

def eval_all(predictor, label):
    env = VoltageControlEnv(predictor, scaler, cols, device, ensemble_is_ensemble=True)
    results = {}
    for algo_name in ALGOS:
        model_path = os.path.join(RL_DIR, 'output', algo_name, 'best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(RL_DIR, 'output', algo_name, 'final_model.pth')
        algo = create_algorithm(algo_name, nf, npots, rl_cfg.ACTION_TRAJECTORY_DIM, device)
        algo.load_model(model_path)
        if hasattr(algo, 'actor'): algo.actor.eval()

        all_errors = []
        is_off = algo_name in ['ddpg', 'td3', 'sac']
        for s in selected:
            state = env.reset(s['past_features'], s['target_voltage'], s['pot_id'])
            for step in range(rl_cfg.MAX_EPISODE_STEPS):
                res = algo.select_action(state)
                action = res if is_off else res[0]
                ns, reward, done, info = env.step(action)
                vp = info['voltage_pred']; vs = info['target_voltage_set']
                if vs is not None and not np.isnan(vp) and not np.isnan(vs):
                    all_errors.append(abs(vp - vs))
                state = ns
                if done: break

        mae = np.mean(all_errors) * 1000
        results[algo_name] = {'mae': mae, 'steps': len(all_errors)}
        print(f'  {ALGO_NAMES[algo_name]:15s} MAE={mae:5.1f} mV')
    return results

# === RUN BOTH ===
print('='*60)
print('PREDICTOR A: Non-augmented (ensemble_baseline_no_aug)')
print('='*60)
pred_a = load_predictor(os.path.join(MODEL_OUT, 'ensemble_baseline_no_aug'))
t0 = time.time()
res_a = eval_all(pred_a, 'A')
t_a = time.time() - t0

print(f'\n{"="*60}')
print('PREDICTOR B: Augmented FIXED (current model/output/)')
print('='*60)
pred_b = load_predictor(MODEL_OUT)
t0 = time.time()
res_b = eval_all(pred_b, 'B')
t_b = time.time() - t0

# === SUMMARY ===
print(f'\n{"="*80}')
print(f'FINAL COMPARISON: Same 8 RL policies, Different predictors')
print(f'{"="*80}')
print(f'{"Algorithm":<18} {"Non-aug":>8} {"Augmented":>8} {"Delta":>8} {"Improve":>8}')
print(f'{"─"*18} {"─"*8} {"─"*8} {"─"*8} {"─"*8}')

total_no_aug, total_aug = [], []
for algo_name in ALGOS:
    m1 = res_a[algo_name]['mae']
    m2 = res_b[algo_name]['mae']
    delta = m1 - m2
    pct = (1 - m2/m1) * 100
    total_no_aug.append(m1)
    total_aug.append(m2)
    print(f'{ALGO_NAMES[algo_name]:<18} {m1:7.1f}mV {m2:7.1f}mV {delta:+6.1f}mV {pct:+6.1f}%')

avg1 = np.mean(total_no_aug)
avg2 = np.mean(total_aug)
print(f'{"─"*18} {"─"*8} {"─"*8} {"─"*8} {"─"*8}')
print(f'{"MEAN":<18} {avg1:7.1f}mV {avg2:7.1f}mV {avg1-avg2:+6.1f}mV {(1-avg2/avg1)*100:+6.1f}%')
print(f'\nTime: {t_a:.0f}s (non-aug) + {t_b:.0f}s (aug) = {t_a+t_b:.0f}s total')
