"""SAME policy × DIFFERENT predictors — direct MAE comparison"""
import os, sys

# Clear all config caches
for k in list(sys.modules.keys()):
    if 'config' in k.lower():
        del sys.modules[k]

# MUST set RL dir first
PROJECT = r'E:\ClaudeCodeWorkplace\2026-5-12-参数优化'
RL_DIR = os.path.join(PROJECT, 'ppo参数优化', 'model')
os.chdir(RL_DIR)
sys.path.insert(0, RL_DIR)
sys.path.insert(1, os.path.join(PROJECT, 'model'))

import config as rl_cfg
import torch, pickle, numpy as np, pandas as pd, glob
from environment import VoltageControlEnv, load_predictor_model
from ppo import MPDPPO

MODEL_OUT = os.path.join(PROJECT, 'model', 'output')
device = torch.device('cuda')
feature_cols = rl_cfg.HIGH_CORR_FEATURES.copy()
num_features = len(feature_cols)
num_pots = len(rl_cfg.TRAIN_POTS) + len(rl_cfg.VAL_POTS) + len(rl_cfg.TEST_POTS)

print(f'Config: {rl_cfg.__file__}')
print(f'Features={num_features}, Pots={num_pots}, ActionDim={rl_cfg.ACTION_TRAJECTORY_DIM}')

# Load scaler
with open(os.path.join(MODEL_OUT, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Build test samples
df = pd.read_excel(rl_cfg.DATA_PATH)
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(['槽号', '日期']).reset_index(drop=True)
for col in feature_cols:
    if col in df.columns:
        df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
        df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
        df[col] = df[col].fillna(df[col].mean())
all_pots = sorted(df['槽号'].unique())
pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

samples = []
for pot_id in rl_cfg.TEST_POTS:
    if pot_id not in df['槽号'].unique():
        continue
    pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
    if len(pot_data) < rl_cfg.INPUT_LEN + rl_cfg.OUTPUT_LEN:
        continue
    pf = pot_data[feature_cols].values
    sv = pot_data['实际设定'].values if '实际设定' in pot_data.columns else pot_data[rl_cfg.TARGET].values
    for i in range(len(pot_data) - rl_cfg.INPUT_LEN - rl_cfg.OUTPUT_LEN + 1):
        samples.append({
            'past_features': pf[i:i + rl_cfg.INPUT_LEN],
            'target_voltage': sv[i + rl_cfg.INPUT_LEN:i + rl_cfg.INPUT_LEN + rl_cfg.OUTPUT_LEN],
            'pot_id': pot_to_idx[pot_id], 'pot_num': pot_id,
        })

# 3 per pot
selected = []
for pot in rl_cfg.TEST_POTS:
    pot_s = [s for s in samples if s['pot_num'] == pot]
    for s in pot_s[-3:]:
        selected.append(s)
print(f'Selected {len(selected)} test samples')

# Load policy ONCE
algo = MPDPPO(num_features, num_pots, rl_cfg.ACTION_TRAJECTORY_DIM, device)
algo.load_model(os.path.join(RL_DIR, 'output', 'mpd_ppo_origbackup', 'best_model.pth'))
algo.actor.eval()
print('Loaded MPD-PPO origbackup policy\n')

def eval_predictor(ckpt_dir, label):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'best_conditional_seed*.pth')))
    predictor = load_predictor_model(None, num_pots, num_features, device,
                                      ensemble=True, ensemble_checkpoint_paths=paths)
    env = VoltageControlEnv(predictor, scaler, feature_cols, device, ensemble_is_ensemble=True)
    all_errors = []
    for s in selected:
        state = env.reset(s['past_features'], s['target_voltage'], s['pot_id'])
        for step in range(rl_cfg.MAX_EPISODE_STEPS):
            action, _ = algo.select_action(state)
            ns, reward, done, info = env.step(action)
            vp = info['voltage_pred']
            vs = info['target_voltage_set']
            if vs is not None and not np.isnan(vp) and not np.isnan(vs):
                all_errors.append(abs(vp - vs))
            state = ns
            if done: break
    mae = np.mean(all_errors) * 1000
    print(f'  {label}: MAE={mae:.1f} mV ({len(all_errors)} steps)')
    return mae

print('='*60)
print('SAME MPD-PPO policy, DIFFERENT predictors')
print('='*60)
mae_base = eval_predictor(os.path.join(MODEL_OUT, 'ensemble_baseline_no_aug'), 'Non-augmented')
mae_aug  = eval_predictor(MODEL_OUT, 'Augmented (FIXED)')
print(f'\nNon-augmented: {mae_base:.1f} mV')
print(f'Augmented:      {mae_aug:.1f} mV')
print(f'Improvement:    {mae_base - mae_aug:.1f} mV ({(1 - mae_aug/mae_base)*100:.1f}%)')
