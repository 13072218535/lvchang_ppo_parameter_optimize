"""
Phase 1: Generate adversarial augmentation data for predictor retraining.

Loads trained RL agents, runs them on training-pot data, collects action sequences,
uses ensemble to assign pseudo-labels (voltage predictions), filters high-uncertainty
(blind spot) samples, and saves the augmented dataset.
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import pandas as pd

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'ppo参数优化', 'model')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, RL_DIR)

from config import *
from data_processor import DataProcessor
from ensemble_predictor import UncertaintyQuantifiedPredictor
from model import LSTMModelWithPotEmbedding, ConditionalVoltagePredictor

# RL imports (from ppo参数优化/model/)
RL_OUTPUT_DIR = os.path.join(RL_DIR, 'output')
RL_ALGO_NAMES = ['mpd_ppo', 'vanilla_ppo', 'a2c', 'taa_ppo', 'taa_ppo_4d', 'ddpg', 'td3', 'sac']


def load_ensemble_predictor(device='cpu'):
    """Load the current 5-member ensemble from checkpoints."""
    output_dir = os.path.join(SCRIPT_DIR, 'output')  # absolute path
    checkpoint_paths = []
    for s in range(5):
        path = os.path.join(output_dir, f'best_conditional_seed{s}.pth')
        if os.path.exists(path):
            checkpoint_paths.append(path)
    if len(checkpoint_paths) < 3:
        raise RuntimeError(f"Need >=3 ensemble checkpoints, found {len(checkpoint_paths)}")

    # Infer input_dim and num_pots from data
    processor = DataProcessor(data_path=DATA_PATH, input_len=INPUT_LEN,
                              output_len=OUTPUT_LEN, split_method='pot')
    df = processor.load_data()
    df = processor.preprocess_data(df)
    _, feature_cols = processor.create_features(df)
    num_features = len(feature_cols)
    num_pots = df['槽号'].nunique()

    predictor = UncertaintyQuantifiedPredictor(
        num_models=5, input_dim=num_features, num_pots=num_pots, device=device
    )
    predictor.load_checkpoints(checkpoint_paths)
    print(f"Loaded {len(checkpoint_paths)} ensemble members")
    return predictor, num_features, num_pots, feature_cols


def load_rl_agents(num_features, num_pots, device='cpu'):
    """Load all trained RL agent models."""
    from algorithms import create_algorithm
    from config import ACTION_TRAJECTORY_DIM

    agents = {}
    for algo_name in RL_ALGO_NAMES:
        model_dir = os.path.join(RL_OUTPUT_DIR, algo_name)
        best_path = os.path.join(model_dir, 'best_model.pth')
        final_path = os.path.join(model_dir, 'final_model.pth')
        model_path = best_path if os.path.exists(best_path) else final_path
        if not os.path.exists(model_path):
            print(f"  Skip {algo_name}: no model found")
            continue

        algo = create_algorithm(algo_name, num_features, num_pots, ACTION_TRAJECTORY_DIM, device)
        algo.load_model(model_path)
        if hasattr(algo, 'actor'):
            algo.actor.eval()
        agents[algo_name] = algo
        print(f"  Loaded {algo_name}: {os.path.basename(model_path)}")
    return agents


def load_training_samples(feature_cols):
    """Load training pot data in the same format as RL training."""
    from config import TRAIN_POTS
    import pickle as pkl

    # Use the RL training data loading approach
    df = pd.read_excel(DATA_PATH)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['槽号', '日期']).reset_index(drop=True)

    scaler_path = os.path.join(os.path.join(SCRIPT_DIR, 'output'), 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pkl.load(f)

    # Preprocess
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
            df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
            df[col] = df[col].fillna(df[col].mean())

    required_cols = ['日期', '槽号'] + [c for c in feature_cols if c in df.columns]
    df = df[required_cols]

    all_pots = sorted(df['槽号'].unique())
    pot_to_idx = {pot: idx for idx, pot in enumerate(all_pots)}

    samples = []
    for pot_id in TRAIN_POTS:  # Only training pots
        if pot_id not in df['槽号'].unique():
            continue
        pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
        if len(pot_data) < INPUT_LEN + OUTPUT_LEN:
            continue
        pot_features = pot_data[feature_cols].values
        pot_set_voltages = pot_data['实际设定'].values if '实际设定' in pot_data.columns else pot_data['工作平均'].values
        for i in range(len(pot_data) - INPUT_LEN - OUTPUT_LEN + 1):
            samples.append({
                'past_features': pot_features[i:i + INPUT_LEN],
                'target_voltage': pot_set_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN],
                'pot_id': pot_to_idx[pot_id],       # 0-indexed for env interaction
                'pot_num': pot_id,                   # actual pot number (1101-1142)
            })
    return samples, scaler


def generate_actions(agents, samples, scaler, ensemble, feature_cols, device='cpu'):
    """
    For each agent and each training sample, generate a 14-day action trajectory.
    Returns list of dicts with past_features, future_actions, predicted_voltage, uncertainty.
    """
    from environment import VoltageControlEnv
    from config import MAX_EPISODE_STEPS, ACTION_ALF_MIN, ACTION_ALF_MAX, ACTION_OUT_MIN, ACTION_OUT_MAX

    env = VoltageControlEnv(ensemble, scaler, feature_cols, device, ensemble_is_ensemble=True)
    results = []

    for algo_name, algo in agents.items():
        print(f"\n  Generating with {algo_name}...")
        is_off_policy = algo_name in ['ddpg', 'td3', 'sac']

        for idx, sample in enumerate(samples):
            state = env.reset(sample['past_features'], sample['target_voltage'], sample['pot_id'])
            all_actions_denorm = []  # collect 14-day actions in physical units

            for step in range(MAX_EPISODE_STEPS):
                result = algo.select_action(state)
                action = result if is_off_policy else result[0]
                next_state, reward, done, info = env.step(action)

                # Collect day-1 action (the one actually executed)
                day1_action = info['day1_action']
                all_actions_denorm.append(day1_action)
                state = next_state
                if done:
                    break

            if len(all_actions_denorm) < MAX_EPISODE_STEPS:
                continue

            # Stack into (14, 2)
            future_actions = np.array(all_actions_denorm)  # (14, 2) in physical units

            # Get ensemble prediction and uncertainty for this action sequence
            # Standardize past features
            past_std = scaler.transform(sample['past_features'])  # (7, 12)

            # Standardize future actions
            alf_idx = feature_cols.index('ALF加料量(实际)') if 'ALF加料量(实际)' in feature_cols else 9
            out_idx = feature_cols.index('实际出铝量') if '实际出铝量' in feature_cols else 10
            alf_mean = scaler.mean_[alf_idx]
            alf_scale = scaler.scale_[alf_idx]
            out_mean = scaler.mean_[out_idx]
            out_scale = scaler.scale_[out_idx]

            fa_std = future_actions.copy()
            fa_std[:, 0] = (fa_std[:, 0] - alf_mean) / alf_scale
            fa_std[:, 1] = (fa_std[:, 1] - out_mean) / out_scale

            _, variance, all_preds = ensemble.predict(
                past_std[None, :, :], fa_std[None, :, :], int(sample['pot_id'])
            )
            mean_var = float(variance.mean())

            # De-standardize predicted voltage
            target_idx = feature_cols.index('工作平均')
            target_mean = scaler.mean_[target_idx]
            target_scale = scaler.scale_[target_idx]
            voltage_pred = all_preds.mean(axis=0)[0] * target_scale + target_mean  # (14,)
            voltage_pred = np.clip(voltage_pred, 2.5, 6.0)

            results.append({
                'past_features': sample['past_features'].copy(),       # (7, 12) raw
                'future_actions': future_actions.copy(),              # (14, 2) raw physical
                'voltage_pred': voltage_pred.copy(),                  # (14,) raw V
                'pot_id': sample['pot_id'],
                'pot_num': sample['pot_num'],
                'algo': algo_name,
                'uncertainty': mean_var,
            })

            if (idx + 1) % 50 == 0:
                print(f"    {algo_name}: {idx+1}/{len(samples)} samples")

    return results


def filter_high_uncertainty(results, p95_threshold, max_per_pot=200):
    """Filter samples with uncertainty above P95 threshold. Also keep a low-uncertainty subset."""
    high_unc = [r for r in results if r['uncertainty'] > p95_threshold]
    low_unc = [r for r in results if r['uncertainty'] < np.median([x['uncertainty'] for x in results])]

    # Cap per-pot samples
    from collections import Counter
    pot_counts_high = Counter()
    filtered_high = []
    for r in sorted(high_unc, key=lambda x: -x['uncertainty']):  # highest uncertainty first
        if pot_counts_high[r['pot_num']] < max_per_pot:
            filtered_high.append(r)
            pot_counts_high[r['pot_num']] += 1

    pot_counts_low = Counter()
    filtered_low = []
    np.random.shuffle(low_unc)
    for r in low_unc:
        if pot_counts_low[r['pot_num']] < max_per_pot // 2:
            filtered_low.append(r)
            pot_counts_low[r['pot_num']] += 1

    return filtered_high, filtered_low


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial augmentation data')
    parser.add_argument('--max_samples_per_pot', type=int, default=5,
                        help='Max training samples per pot for agent exploration')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--p95_threshold', type=float, default=None,
                        help='P95 uncertainty threshold (auto-loaded if not specified)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load ensemble predictor
    print("\n1. Loading ensemble predictor...")
    predictor, num_features, num_pots, feature_cols = load_ensemble_predictor(device)

    # Load uncertainty threshold from calibration
    threshold_path = os.path.join(os.path.join(SCRIPT_DIR, 'output'), 'uncertainty_threshold.json')
    if os.path.exists(threshold_path):
        import json
        with open(threshold_path) as f:
            thresholds = json.load(f)
        p95 = args.p95_threshold if args.p95_threshold else thresholds.get('P95', 0.01)
        print(f"   P95 uncertainty threshold: {p95:.6f}")
    else:
        p95 = args.p95_threshold if args.p95_threshold else 0.01
        print(f"   Using default threshold: {p95:.6f}")

    # 2. Load RL agents
    print("\n2. Loading RL agents...")
    agents = load_rl_agents(num_features, num_pots, device)
    print(f"   Loaded {len(agents)} agents")

    # 3. Load training samples
    print("\n3. Loading training samples...")
    all_samples, scaler = load_training_samples(feature_cols)
    print(f"   Total training samples: {len(all_samples)}")

    # Sample a subset for efficiency
    np.random.seed(42)
    n_samples = min(args.max_samples_per_pot * 30, len(all_samples))  # ~5 per pot × 30 pots
    selected = np.random.choice(len(all_samples), n_samples, replace=False)
    train_samples = [all_samples[i] for i in selected]
    print(f"   Selected {len(train_samples)} samples for exploration")

    # 4. Generate agent exploration data
    print("\n4. Generating agent exploration data...")
    results = generate_actions(agents, train_samples, scaler, predictor, feature_cols, device)
    print(f"\n   Generated {len(results)} total samples")

    # 5. Filter high-uncertainty samples
    print("\n5. Filtering high-uncertainty samples...")
    high_unc, low_unc = filter_high_uncertainty(results, p95, max_per_pot=200)
    print(f"   High uncertainty (>P95): {len(high_unc)} samples")
    print(f"   Low uncertainty (<P50):  {len(low_unc)} samples")

    # Uncertainty distribution stats
    all_unc = [r['uncertainty'] for r in results]
    print(f"\n   Uncertainty stats: mean={np.mean(all_unc):.6f}, "
          f"median={np.median(all_unc):.6f}, max={np.max(all_unc):.6f}")

    # 6. Build augmented dataset
    print("\n6. Building augmented dataset...")
    augmented = high_unc + low_unc  # mix both for training

    X_list = [r['past_features'] for r in augmented]             # (7, 12) raw
    y_list = [r['voltage_pred'] for r in augmented]              # (14,) raw V
    fa_list = [r['future_actions'] for r in augmented]           # (14, 2) raw
    pid_list = [r['pot_num'] for r in augmented]                 # ACTUAL pot numbers (1101-1142)
    unc_list = [r['uncertainty'] for r in augmented]

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y_voltage': np.array(y_list, dtype=np.float32),
        'future_actions': np.array(fa_list, dtype=np.float32),
        'pot_ids': np.array(pid_list, dtype=np.int32),
        'uncertainty': np.array(unc_list, dtype=np.float32),
    }
    print(f"   Dataset shapes: X={dataset['X'].shape}, y={dataset['y_voltage'].shape}, "
          f"actions={dataset['future_actions'].shape}")

    # 7. Save
    save_path = os.path.join(os.path.join(SCRIPT_DIR, 'output'), 'adversarial_augmented.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\n7. Saved augmented dataset to: {save_path}")
    print(f"   Total augmented samples: {len(augmented)} "
          f"(high_unc={len(high_unc)}, low_unc={len(low_unc)})")

    # Quick stats
    print(f"\nPer-algo contribution:")
    algo_counts = {}
    for r in augmented:
        algo_counts[r['algo']] = algo_counts.get(r['algo'], 0) + 1
    for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
        print(f"  {algo}: {count}")


if __name__ == '__main__':
    main()
