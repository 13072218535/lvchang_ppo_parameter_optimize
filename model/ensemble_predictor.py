"""
Deep Ensemble Predictor with Uncertainty Quantification.

Wraps N ConditionalVoltagePredictor instances trained with different random seeds.
At inference time returns mean prediction and per-sample variance across members.
The variance captures epistemic uncertainty — when an action sequence lies outside
the training distribution, ensemble members disagree, producing high variance.

Also includes MCDropoutPredictor as a lightweight single-model alternative.
"""
import torch
import numpy as np
import os

from model import LSTMModelWithPotEmbedding, ConditionalVoltagePredictor
from config import HIDDEN_DIM, NUM_LAYERS, DROPOUT, POT_EMBED_DIM, OUTPUT_LEN


class UncertaintyQuantifiedPredictor:
    """Ensemble-based uncertainty-aware conditional voltage predictor."""

    def __init__(self, num_models=5, input_dim=12, num_pots=42,
                 future_action_dim=2, future_len=OUTPUT_LEN, cond_hidden=32,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT, device='cpu'):
        self.num_models = num_models
        self.input_dim = input_dim
        self.num_pots = num_pots
        self.device = torch.device(device)
        self.models = []
        self._build_ensemble(input_dim, num_pots, future_action_dim,
                            future_len, cond_hidden, hidden_dim, dropout)

    def _build_ensemble(self, input_dim, num_pots, future_action_dim,
                         future_len, cond_hidden, hidden_dim, dropout):
        for _ in range(self.num_models):
            base_model = LSTMModelWithPotEmbedding(
                input_dim=input_dim, num_pots=num_pots,
                pot_embed_dim=POT_EMBED_DIM, hidden_dim=hidden_dim,
                num_layers=NUM_LAYERS, output_len=future_len, dropout=dropout
            ).to(self.device)
            predictor = ConditionalVoltagePredictor(
                base_model=base_model, future_action_dim=future_action_dim,
                future_len=future_len, cond_hidden=cond_hidden,
                hidden_dim=hidden_dim, dropout=dropout
            ).to(self.device)
            predictor.eval()
            self.models.append(predictor)

    def load_checkpoints(self, checkpoint_paths):
        """Load pretrained weights for each ensemble member."""
        assert len(checkpoint_paths) == self.num_models, \
            f"Expected {self.num_models} checkpoints, got {len(checkpoint_paths)}"
        for i, path in enumerate(checkpoint_paths):
            if not os.path.exists(path):
                print(f"  [WARN] Checkpoint not found: {path}, member {i} stays at random init")
                continue
            ckpt = torch.load(path, map_location=self.device)
            try:
                self.models[i].load_state_dict(ckpt)
                print(f"  Member {i} loaded: {os.path.basename(path)}")
            except RuntimeError:
                self._partial_load(self.models[i], ckpt, i, path)
            self.models[i].eval()

    def _partial_load(self, model, checkpoint, idx, path):
        model_dict = model.state_dict()
        matched = {}
        skipped = []
        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"  Member {idx} partial load: {len(matched)}/{len(model_dict)} keys matched, "
              f"{len(skipped)} skipped ({os.path.basename(path)})")

    def predict(self, past_features, future_actions, pot_id):
        """
        Args:
            past_features: numpy (B, 7, input_dim) — standardized
            future_actions: numpy (B, 14, 2) — standardized
            pot_id: int or numpy array of ints
        Returns:
            mean: numpy (B, 14)
            variance: numpy (B, 14) — sample variance across ensemble
            all_preds: numpy (N, B, 14)
        """
        B = past_features.shape[0]
        all_preds = np.zeros((self.num_models, B, OUTPUT_LEN), dtype=np.float32)

        past_tensor = torch.FloatTensor(past_features).to(self.device)
        future_tensor = torch.FloatTensor(future_actions).to(self.device)
        if isinstance(pot_id, (int, np.integer)):
            pot_tensor = torch.LongTensor([pot_id] * B).to(self.device)
        else:
            pot_tensor = torch.LongTensor(np.asarray(pot_id)).to(self.device)

        with torch.no_grad():
            for i, model in enumerate(self.models):
                pred = model(past_tensor, future_tensor, pot_tensor)
                all_preds[i] = pred.cpu().numpy()

        mean = all_preds.mean(axis=0)
        variance = all_preds.var(axis=0, ddof=1)
        variance = np.maximum(variance, 1e-8)
        return mean, variance, all_preds

    def predict_with_stats(self, past_features, future_actions, pot_id):
        """Returns (mean, variance, per_day_std, all_preds)."""
        mean, variance, all_preds = self.predict(past_features, future_actions, pot_id)
        per_day_std = np.sqrt(variance)
        return mean, variance, per_day_std, all_preds

    def estimate_threshold(self, train_loader, percentile=95):
        """Calibrate uncertainty threshold from training data distribution."""
        uncertainties = []
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            X, y, future_actions, pot_ids = batch
            _, variance, _ = self.predict(
                X.numpy(), future_actions.numpy(), pot_ids.numpy()
            )
            mean_var = variance.mean(axis=1)
            uncertainties.extend(mean_var.tolist())
        thresholds = {}
        for p in [50, 75, 90, 95, 99]:
            thresholds[f'P{p}'] = float(np.percentile(uncertainties, p))
        thresholds['mean'] = float(np.mean(uncertainties))
        thresholds['max'] = float(np.max(uncertainties))
        print(f"Uncertainty calibration (n={len(uncertainties)} samples):")
        for k, v in thresholds.items():
            print(f"  {k}: {v:.6f}")
        return thresholds

    def to(self, device):
        self.device = torch.device(device)
        for m in self.models:
            m.to(self.device)

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train()


class MCDropoutPredictor:
    """Lightweight uncertainty via MC Dropout — single model, M stochastic passes."""

    def __init__(self, model, num_samples=30, device='cpu'):
        self.model = model
        self.num_samples = num_samples
        self.device = torch.device(device)

    def predict(self, past_features, future_actions, pot_id):
        B = past_features.shape[0]
        all_preds = np.zeros((self.num_samples, B, OUTPUT_LEN), dtype=np.float32)

        past_tensor = torch.FloatTensor(past_features).to(self.device)
        future_tensor = torch.FloatTensor(future_actions).to(self.device)
        if isinstance(pot_id, (int, np.integer)):
            pot_tensor = torch.LongTensor([pot_id] * B).to(self.device)
        else:
            pot_tensor = torch.LongTensor(np.asarray(pot_id)).to(self.device)

        self.model.train()  # keep dropout active
        with torch.no_grad():
            for s in range(self.num_samples):
                pred = self.model(past_tensor, future_tensor, pot_tensor)
                all_preds[s] = pred.cpu().numpy()
        self.model.eval()

        mean = all_preds.mean(axis=0)
        variance = all_preds.var(axis=0, ddof=1)
        variance = np.maximum(variance, 1e-8)
        return mean, variance, all_preds
