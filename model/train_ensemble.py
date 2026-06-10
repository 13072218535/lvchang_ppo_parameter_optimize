"""
Train an ensemble of ConditionalVoltagePredictor models with different random seeds
for uncertainty quantification.

Usage:
  # Phase 1: Train base LSTMs
  python train_ensemble.py --phase base --num_seeds 5 --device cpu

  # Phase 2: Train conditional predictors (loads base models from phase 1)
  python train_ensemble.py --phase conditional --num_seeds 5 --device cpu

  # Phase 3: Calibrate uncertainty threshold
  python train_ensemble.py --phase calibrate --num_seeds 5 --device cpu
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
import argparse
import json
import glob as glob_module

from data_processor import DataProcessor
from model import LSTMModel, LSTMModelWithPotEmbedding, ConditionalVoltagePredictor
from ensemble_predictor import UncertaintyQuantifiedPredictor
from config import *
from train import (train_epoch, validate_epoch, test_evaluate,
                   calculate_metrics, evaluate_by_day)


def train_single_seed(seed, device, use_conditional=True, augmented_data_path=None):
    """
    Train one model with a specific random seed.
    Returns metrics dict and saves checkpoint.
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Training with seed={seed}, conditional={use_conditional}")
    if augmented_data_path:
        print(f"Augmented data: {augmented_data_path}")
    print(f"{'='*60}")

    processor = DataProcessor(data_path=DATA_PATH, input_len=INPUT_LEN,
                              output_len=OUTPUT_LEN, split_method='pot')
    train_loader, val_loader, test_loader, num_pots, feature_cols = \
        processor.process(use_future_actions=use_conditional,
                         augmented_data_path=augmented_data_path)

    num_features = train_loader.dataset.X.shape[-1]

    # Determine checkpoint suffix
    suffix = 'conditional' if use_conditional else 'model'
    model_save_path = os.path.join(OUTPUT_DIR, f'best_{suffix}_seed{seed}.pth')
    final_model_path = os.path.join(OUTPUT_DIR, f'final_{suffix}_seed{seed}.pth')

    if use_conditional:
        base_model = LSTMModelWithPotEmbedding(
            input_dim=num_features, num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, output_len=OUTPUT_LEN, dropout=DROPOUT
        ).to(device)

        # Load pretrained base model for this seed
        base_path = os.path.join(OUTPUT_DIR, f'best_model_seed{seed}.pth')
        if os.path.exists(base_path):
            try:
                base_model.load_state_dict(torch.load(base_path, map_location=device))
                print(f"Loaded base model: {base_path}")
                for param in base_model.parameters():
                    param.requires_grad = True
            except RuntimeError as e:
                print(f"WARN: Base model architecture mismatch: {e}")
                print("Base model will start from random init")
        else:
            print(f"WARN: Base model not found at {base_path}, using random init")

        model = ConditionalVoltagePredictor(
            base_model=base_model, future_action_dim=2,
            future_len=OUTPUT_LEN, cond_hidden=32,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT
        ).to(device)
    else:
        model = LSTMModelWithPotEmbedding(
            input_dim=num_features, num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, output_len=OUTPUT_LEN, dropout=DROPOUT
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR,
                                  patience=REDUCE_LR_PATIENCE)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    use_pot_embedding = True

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 device, use_pot_embedding, use_conditional)
        val_loss = validate_epoch(model, val_loader, criterion, device,
                                  use_pot_embedding, use_conditional)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}: train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"best={best_val_loss:.6f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), final_model_path)

    # Test evaluation
    y_true, y_pred = test_evaluate(model, test_loader, device,
                                   use_pot_embedding, use_conditional)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics = calculate_metrics(y_true_flat, y_pred_flat)

    result = {
        'seed': seed,
        'best_val_loss': float(best_val_loss),
        'test_mae': float(metrics['MAE']),
        'test_rmse': float(metrics['RMSE']),
        'test_r2': float(metrics['R2']),
        'epochs_trained': len(train_losses),
        'model_path': model_save_path,
    }
    return result


def calibrate_threshold(num_seeds, device):
    """Load all ensemble checkpoints and calibrate uncertainty threshold."""
    print(f"\n{'='*60}")
    print("Calibrating uncertainty threshold from training data")
    print(f"{'='*60}")

    # Load training data
    processor = DataProcessor(data_path=DATA_PATH, input_len=INPUT_LEN,
                              output_len=OUTPUT_LEN, split_method='pot')
    train_loader, _, _, num_pots, feature_cols = \
        processor.process(use_future_actions=True)
    num_features = train_loader.dataset.X.shape[-1]

    # Load ensemble
    checkpoint_paths = []
    for s in range(num_seeds):
        path = os.path.join(OUTPUT_DIR, f'best_conditional_seed{s}.pth')
        checkpoint_paths.append(path)

    predictor = UncertaintyQuantifiedPredictor(
        num_models=num_seeds, input_dim=num_features,
        num_pots=num_pots, device=device
    )
    predictor.load_checkpoints(checkpoint_paths)

    # Calibrate on training set subset (max 500 samples to keep it fast)
    thresholds = predictor.estimate_threshold(train_loader, percentile=95)

    # Save
    threshold_path = os.path.join(OUTPUT_DIR, 'uncertainty_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved to: {threshold_path}")

    # Quick OOD check
    print(f"\n{'='*60}")
    print("OOD sensitivity check")
    print(f"{'='*60}")
    try:
        sample_batch = next(iter(train_loader))
        X_sample = sample_batch[0][:1].numpy()
        fa_sample = sample_batch[2][:1].numpy()
        pi_sample = sample_batch[3][:1].numpy()

        # In-distribution
        _, var_id, _ = predictor.predict(X_sample, fa_sample, pi_sample)

        # Random OOD actions
        fa_random = np.random.randn(*fa_sample.shape).astype(np.float32) * 2.0
        fa_random = np.clip(fa_random, -4, 4)
        _, var_ood, _ = predictor.predict(X_sample, fa_random, pi_sample)

        print(f"  In-distribution mean variance: {var_id.mean():.6f}")
        print(f"  Random OOD mean variance:     {var_ood.mean():.6f}")
        print(f"  Ratio OOD/ID:                 {var_ood.mean() / max(var_id.mean(), 1e-8):.2f}x")
    except Exception as e:
        print(f"  OOD check failed: {e}")

    return thresholds


def main():
    parser = argparse.ArgumentParser(description='Train ensemble of conditional predictors')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['base', 'conditional', 'calibrate'],
                        help='Training phase')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of ensemble members (default: 5)')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting seed index (default: 0)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--use_augmented', action='store_true',
                        help='Include adversarial augmented data in training')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")

    aug_path = os.path.join(OUTPUT_DIR, 'adversarial_augmented.pkl') if args.use_augmented else None
    if args.use_augmented:
        print(f"Augmented data: {aug_path} (exists={os.path.exists(aug_path)})")

    if args.phase == 'calibrate':
        calibrate_threshold(args.num_seeds, device)
        return

    use_conditional = (args.phase == 'conditional')
    results = []

    for s in range(args.start_seed, args.start_seed + args.num_seeds):
        result = train_single_seed(s, device, use_conditional=use_conditional,
                                   augmented_data_path=aug_path)
        results.append(result)
        print(f"  Seed {s}: val_loss={result['best_val_loss']:.6f}, "
              f"test_mae={result['test_mae']:.6f}, test_rmse={result['test_rmse']:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Ensemble Training Summary ({args.phase})")
    print(f"{'='*60}")
    maes = [r['test_mae'] for r in results]
    print(f"  Individual MAEs: {[f'{m:.6f}' for m in maes]}")
    print(f"  Mean MAE: {np.mean(maes):.6f} ± {np.std(maes):.6f}")

    # Save summary
    suffix = 'conditional' if use_conditional else 'model'
    summary_path = os.path.join(OUTPUT_DIR, f'ensemble_{suffix}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
