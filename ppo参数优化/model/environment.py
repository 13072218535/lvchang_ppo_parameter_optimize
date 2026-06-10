"""
MPD-PPO仿真环境类
封装条件电压预测模型，实现MDP转移
"""
import sys
import numpy as np
import torch
import torch.nn as nn

from model import LSTMModelWithPotEmbedding, ConditionalVoltagePredictor
from config import *


class VoltageControlEnv:
    """电压控制仿真环境"""
    
    def __init__(self, predictor_model, scaler, feature_cols, device='cpu',
                 uncertainty_config=None, ensemble_is_ensemble=False):
        """
        参数:
            predictor_model: 训练好的条件电压预测模型（单模型）或 UncertaintyQuantifiedPredictor（ensemble模式）
            scaler: 数据标准化器
            feature_cols: 特征列名列表（用于索引映射）
            device: 设备（cpu或cuda）
            uncertainty_config: dict with keys 'lambda', 'use_threshold', 'threshold'
            ensemble_is_ensemble: True if predictor_model is an ensemble wrapper
        """
        self.predictor = predictor_model
        self.scaler = scaler
        self.device = device
        self.feature_cols = feature_cols
        self.ensemble_is_ensemble = ensemble_is_ensemble

        # Uncertainty quantification config
        if uncertainty_config is None:
            uncertainty_config = {}
        self.unc_lambda = uncertainty_config.get('lambda', 1.0)
        self.unc_use_threshold = uncertainty_config.get('use_threshold', True)
        self.unc_threshold = uncertainty_config.get('threshold', None)

        self.pot_embed_dim = POT_EMBED_DIM
        self.hidden_dim = HIDDEN_DIM

        self._build_feature_indices()

        self.alf_min = ACTION_ALF_MIN
        self.alf_max = ACTION_ALF_MAX
        self.out_min = ACTION_OUT_MIN
        self.out_max = ACTION_OUT_MAX

        self.alf_max_change = ACTION_ALF_MAX_CHANGE
        self.out_max_change = ACTION_OUT_MAX_CHANGE

        self.state_dim = None

        self.current_state = None
        self.current_step = 0
        self.past_features = None
        self.target_voltage = None
        self.pot_id = None
        self.last_action = None
        self.cumulative_error = 0.0
        self._smooth_weight = REWARD_SMOOTH_VIOLATION_WEIGHT
        self._bound_weight = REWARD_BOUND_PENALTY  # 可被外部覆盖

    def set_smoothness_weight(self, weight):
        """动态调整平滑惩罚权重（off-policy课程学习用）"""
        self._smooth_weight = weight

    def set_bound_penalty_weight(self, weight):
        """动态调整边界惩罚权重（TAA-PPO-4D专用：4D架构无法避免边界）"""
        self._bound_weight = weight
    
    def _build_feature_indices(self):
        """构建特征列索引映射（仅原始特征，无统计/衍生特征）"""
        self.feature_indices = {
            'target': self.feature_cols.index('工作平均') if '工作平均' in self.feature_cols else -1,
            'alf_actual': self.feature_cols.index('ALF加料量(实际)') if 'ALF加料量(实际)' in self.feature_cols else -1,
            'out_actual': self.feature_cols.index('实际出铝量') if '实际出铝量' in self.feature_cols else -1,
        }
        # 缓存非控制特征索引（在_update_state中沿用前一天值）
        self._carryover_indices = []
        for feat in self.feature_cols:
            if feat not in ['工作平均', 'ALF加料量(实际)', '实际出铝量', '平均电压', '铝水平']:
                idx = self.feature_cols.index(feat)
                self._carryover_indices.append(idx)
    
    def _normalize_action(self, action):
        """将动作从[-1,1]范围映射到实际范围"""
        alf = (action[0] + 1) / 2 * (self.alf_max - self.alf_min) + self.alf_min
        out = (action[1] + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return np.array([alf, out])
    
    def _denormalize_action(self, action):
        """将动作从实际范围映射到[-1,1]范围"""
        alf = (action[0] - self.alf_min) / (self.alf_max - self.alf_min) * 2 - 1
        out = (action[1] - self.out_min) / (self.out_max - self.out_min) * 2 - 1
        return np.array([alf, out])
    
    def _clip_action(self, action):
        """裁剪动作为有效范围"""
        action[0] = np.clip(action[0], self.alf_min, self.alf_max)
        action[1] = np.clip(action[1], self.out_min, self.out_max)
        return action
    
    def _reshape_trajectory(self, action_trajectory):
        """
        将28维动作轨迹重塑为(14, 2)的动作序列
        参数:
            action_trajectory: (28,) 数组，[alf1, out1, alf2, out2, ..., alf14, out14]
        返回:
            future_actions: (14, 2) 数组
        """
        if len(action_trajectory) != ACTION_TRAJECTORY_DIM:
            raise ValueError(f"Expected {ACTION_TRAJECTORY_DIM}-dim trajectory, got {len(action_trajectory)}")
        future_actions = action_trajectory.reshape(OUTPUT_LEN, 2)
        return future_actions
    
    def _predict_voltage(self, past_features, future_actions, pot_id):
        """调用条件预测模型预测电压（处理标准化/反标准化）。
        Returns (voltage_pred, uncertainty) where uncertainty is (14,) variance per day
        or None for single-model mode.
        """
        self.predictor.eval()
        with torch.no_grad():
            # past_features标准化（与训练时一致）
            standardized_features = self.scaler.transform(past_features)
            past_tensor = torch.FloatTensor(standardized_features).unsqueeze(0).to(self.device)

            # future_actions标准化（修复：训练时用标准化值，仿真时必须一致）
            alf_idx = self.feature_indices['alf_actual']
            out_idx = self.feature_indices['out_actual']
            alf_mean = self.scaler.mean_[alf_idx]
            alf_scale = self.scaler.scale_[alf_idx]
            out_mean = self.scaler.mean_[out_idx]
            out_scale = self.scaler.scale_[out_idx]

            future_actions_std = future_actions.copy()
            future_actions_std[:, 0] = (future_actions_std[:, 0] - alf_mean) / alf_scale
            future_actions_std[:, 1] = (future_actions_std[:, 1] - out_mean) / out_scale
            future_actions_tensor = torch.FloatTensor(future_actions_std).unsqueeze(0).to(self.device)

            pot_id_tensor = torch.LongTensor([pot_id]).to(self.device)

            if self.ensemble_is_ensemble:
                # Ensemble: get mean prediction + per-day variance
                mean, variance, _ = self.predictor.predict(
                    standardized_features[None, :, :],
                    future_actions_std[None, :, :],
                    int(pot_id)
                )
                voltage_pred = mean[0]        # (14,)
                uncertainty = variance[0]     # (14,) per-day variance
            else:
                # Legacy single-model path
                voltage_pred = self.predictor(past_tensor, future_actions_tensor, pot_id_tensor)
                voltage_pred = voltage_pred.cpu().numpy()[0]
                uncertainty = None

            # 反标准化：模型输出是标准化值(μ≈0,σ≈1)，转为原始电压(V)
            target_idx = self.feature_indices['target']
            target_mean = self.scaler.mean_[target_idx]
            target_scale = self.scaler.scale_[target_idx]
            voltage_pred = voltage_pred * target_scale + target_mean

            # 钳制到物理合理范围
            voltage_pred = np.clip(voltage_pred, 2.5, 6.0)
            return voltage_pred, uncertainty
    
    def _calculate_reward(self, voltage_pred, target_voltage, action_trajectory,
                           uncertainty=None):
        """
        计算多步加权奖励函数
        R_acc = Σ_{i=0}^{13} w_i * ACC_WEIGHT * exp(-κ * (v̂_i - v_set_i)²)
        R_t = R_acc * exp(-λ * mean_var) + P_smooth + P_bound
        参数:
            voltage_pred: (14,) 预测电压序列
            target_voltage: (14,) 目标电压序列（设定电压）
            action_trajectory: (14, 2) 14天动作轨迹（未归一化）
            uncertainty: (14,) per-day ensemble variance, or None for single-model mode
        """
        # 多步加权精度奖励
        errors = np.abs(voltage_pred - target_voltage)
        time_weights = np.array(REWARD_TIME_WEIGHTS[:len(voltage_pred)])
        R_acc = REWARD_ACC_WEIGHT * np.sum(
            time_weights * np.exp(-REWARD_ACC_KAPPA * (errors ** 2))
        )

        # 平滑惩罚（检查相邻天动作变化，归一化到比例空间）
        # 极端违反：|Δ| > 2×上限时施加额外乘数，防止Agent牺牲平滑换取微小精度提升
        alf_range = self.alf_max - self.alf_min
        out_range = self.out_max - self.out_min
        smooth_penalty = 0.0
        for i in range(1, len(action_trajectory)):
            alf_diff = abs(action_trajectory[i, 0] - action_trajectory[i-1, 0])
            out_diff = abs(action_trajectory[i, 1] - action_trajectory[i-1, 1])
            alf_violation = max(0, alf_diff - self.alf_max_change) / alf_range
            out_violation = max(0, out_diff - self.out_max_change) / out_range
            if alf_violation > 0:
                penalty = self._smooth_weight * alf_violation
                if alf_diff > 2 * self.alf_max_change:
                    penalty *= REWARD_SMOOTH_EXTREME_MULTIPLIER
                smooth_penalty -= penalty
            if out_violation > 0:
                penalty = self._smooth_weight * out_violation
                if out_diff > 2 * self.out_max_change:
                    penalty *= REWARD_SMOOTH_EXTREME_MULTIPLIER
                smooth_penalty -= penalty
        # 加上与上一步最后动作的平滑约束（如果存在）
        if self.last_action is not None:
            alf_diff = abs(action_trajectory[0, 0] - self.last_action[0])
            out_diff = abs(action_trajectory[0, 1] - self.last_action[1])
            alf_violation = max(0, alf_diff - self.alf_max_change) / alf_range
            out_violation = max(0, out_diff - self.out_max_change) / out_range
            if alf_violation > 0:
                penalty = self._smooth_weight * alf_violation
                if alf_diff > 2 * self.alf_max_change:
                    penalty *= REWARD_SMOOTH_EXTREME_MULTIPLIER
                smooth_penalty -= penalty
            if out_violation > 0:
                penalty = self._smooth_weight * out_violation
                if out_diff > 2 * self.out_max_change:
                    penalty *= REWARD_SMOOTH_EXTREME_MULTIPLIER
                smooth_penalty -= penalty

        # 边界接近惩罚（替代旧的≥/≤边界检查，改为接近边界即惩罚）
        # 动作进入[min, min+margin*range]或[max-margin*range, max]时线性惩罚
        # 越靠近边界惩罚越大，在边界处达到-REWARD_BOUND_PENALTY
        P_bound = 0.0
        alf_margin = REWARD_BOUND_MARGIN * alf_range
        out_margin = REWARD_BOUND_MARGIN * out_range
        for i in range(len(action_trajectory)):
            a = action_trajectory[i]
            # ALF边界接近惩罚
            if a[0] < self.alf_min + alf_margin:
                ratio = (self.alf_min + alf_margin - a[0]) / alf_margin
                P_bound -= self._bound_weight * ratio
            elif a[0] > self.alf_max - alf_margin:
                ratio = (a[0] - (self.alf_max - alf_margin)) / alf_margin
                P_bound -= self._bound_weight * ratio
            # OUT边界接近惩罚
            if a[1] < self.out_min + out_margin:
                ratio = (self.out_min + out_margin - a[1]) / out_margin
                P_bound -= self._bound_weight * ratio
            elif a[1] > self.out_max - out_margin:
                ratio = (a[1] - (self.out_max - out_margin)) / out_margin
                P_bound -= self._bound_weight * ratio

        # 不确定性惩罚（乘法折扣形式）：
        # P_unc将R_acc按比例折扣：越OOD→折扣越大，量级与R_acc天然匹配
        P_unc = 0.0
        if uncertainty is not None:
            mean_var = float(np.mean(uncertainty))
            discount = float(np.exp(-self.unc_lambda * mean_var))
            P_unc = R_acc * (discount - 1.0)  # 负值，表示被扣掉的R_acc部分
            R_acc_discounted = R_acc * discount
        else:
            R_acc_discounted = R_acc

        reward = R_acc_discounted + smooth_penalty + P_bound

        if np.isinf(reward) or np.isnan(reward):
            reward = R_acc

        reward_components = {'R_acc': R_acc, 'P_smooth': smooth_penalty,
                             'P_bound': P_bound, 'P_unc': P_unc}
        return reward, errors[0], reward_components

    def _update_state(self, future_actions_full, voltage_pred, prev_features):
        """
        更新状态：滑动时间窗口，构造新的过去7天状态。
        修复违反3：用14天加权动作替代仅day-1动作，使s'编码完整计划信息。

        参数:
            future_actions_full: (14, 2) 完整14天动作轨迹（未归一化）
            voltage_pred: 预测的电压序列（完整14天）
            prev_features: 更新前的特征窗口 (7, feature_dim)
        """
        new_past_features = prev_features.copy()

        new_past_features = np.concatenate([new_past_features[1:], new_past_features[-1:]], axis=0)

        last_day_idx = -1
        second_last_idx = -2

        # 更新目标变量（工作平均电压）为预测值，钳制防止正反馈爆炸
        if self.feature_indices['target'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['target']] = np.clip(voltage_pred[0], 2.5, 6.0)

        # 修复C：14天加权动作更新控制变量
        # 权重与reward时间衰减一致，day-1权重最高，编码"意图"信息到状态
        time_w = np.array(REWARD_TIME_WEIGHTS, dtype=np.float64)
        w_sum = np.sum(time_w)
        weighted_alf = np.sum(time_w * future_actions_full[:, 0]) / w_sum
        weighted_out = np.sum(time_w * future_actions_full[:, 1]) / w_sum

        if self.feature_indices['alf_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['alf_actual']] = weighted_alf
        if self.feature_indices['out_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['out_actual']] = weighted_out

        day1_action = future_actions_full[0]  # day-1用于经验更新公式

        # === 非控制特征处理（基于day-1动作的经验公式）===
        prev_alf = prev_features[-1, self.feature_indices['alf_actual']] if self.feature_indices['alf_actual'] >= 0 else day1_action[0]
        prev_out = prev_features[-1, self.feature_indices['out_actual']] if self.feature_indices['out_actual'] >= 0 else day1_action[1]
        delta_out = day1_action[1] - prev_out

        # 铝水平：出铝量增加 → 铝水平降低（经验关系）
        al_idx = self.feature_cols.index('铝水平') if '铝水平' in self.feature_cols else -1
        if al_idx >= 0 and prev_out > 0:
            al_change = EMPIRICAL_AL_OUT_RATIO * (delta_out / prev_out) * new_past_features[second_last_idx, al_idx]
            new_past_features[last_day_idx, al_idx] = np.clip(
                new_past_features[second_last_idx, al_idx] + al_change, 0, 100)

        # 平均电压：直接用预测电压更新，钳制防止正反馈爆炸
        avg_v_idx = self.feature_cols.index('平均电压') if '平均电压' in self.feature_cols else -1
        if avg_v_idx >= 0:
            new_past_features[last_day_idx, avg_v_idx] = np.clip(voltage_pred[0], 3.0, 5.0)

        # 其余非控制特征（Fe含量、槽龄、电解质水平等）：沿用前一天值
        for idx in self._carryover_indices:
            new_past_features[last_day_idx, idx] = new_past_features[second_last_idx, idx]

        # 更新累计误差
        self.cumulative_error += abs(voltage_pred[0] - self.target_voltage[0])

        # 更新目标电压序列（滑动一天，用末值填充保持14元素长度）
        self.target_voltage = np.append(self.target_voltage[1:], self.target_voltage[-1])

        return new_past_features
    
    def reset(self, past_features, target_voltage, pot_id):
        """
        重置环境到初始状态
        参数:
            past_features: (7, feature_dim) 过去7天的特征
            target_voltage: (14,) 未来14天的设定电压序列
            pot_id: 槽号索引
        """
        self.past_features = past_features.copy()
        self.target_voltage = target_voltage.copy()
        self.pot_id = pot_id
        self.current_step = 0
        self.last_action = None
        self.cumulative_error = 0.0
        
        # 计算状态维度
        self.state_dim = past_features.shape[1] + len(target_voltage) + 1
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态向量（满足马尔可夫性）
        包含: 7天历史窗口 + 目标电压 + 上次动作 + 累积误差 + 槽号
        """
        past_features_flat = self.past_features.flatten()
        # last_action编码跨episode平滑约束所需信息（修复违反1）
        last_act = np.zeros(2, dtype=np.float32) if self.last_action is None else self.last_action.astype(np.float32)
        # cumulative_error编码终止条件（修复违反2）
        cum_err = np.array([self.cumulative_error / max(MAX_CUMULATIVE_ERROR, 1e-6)], dtype=np.float32)
        state = np.concatenate([past_features_flat, self.target_voltage, last_act, cum_err, [self.pot_id]])
        return state
    
    def step(self, action_trajectory):
        """
        执行一步动作（滚动时域模式）
        参数:
            action_trajectory: (28,) 归一化到[-1,1]的14天×2动作轨迹
                              [alf1, out1, alf2, out2, ..., alf14, out14]
        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否终止
            info: 附加信息
        """
        future_actions = self._reshape_trajectory(action_trajectory)

        future_actions_denorm = np.zeros_like(future_actions)
        for i in range(len(future_actions)):
            denormed = self._normalize_action(future_actions[i])
            future_actions_denorm[i] = self._clip_action(denormed)

        voltage_pred, uncertainty = self._predict_voltage(self.past_features, future_actions_denorm, self.pot_id)

        reward, day1_error, reward_components = self._calculate_reward(
            voltage_pred, self.target_voltage, future_actions_denorm,
            uncertainty=uncertainty)

        # 在滑动前保存当前天的目标电压（用于info输出）
        current_target = self.target_voltage[0] if len(self.target_voltage) > 0 else None
        current_target_full = self.target_voltage.copy()

        prev_features = self.past_features.copy()
        day1_action = future_actions_denorm[0]
        # 修复C：传递完整14天轨迹给_update_state
        self.past_features = self._update_state(future_actions_denorm, voltage_pred, prev_features)

        self.current_step += 1

        done = False
        if self.current_step >= MAX_EPISODE_STEPS:
            done = True
        if self.cumulative_error >= MAX_CUMULATIVE_ERROR:
            done = True

        self.last_action = day1_action.copy()

        next_state = self._get_state()

        info = {
            'voltage_pred': voltage_pred[0],
            'voltage_pred_full': voltage_pred.copy(),
            'target_voltage_set': current_target,
            'target_voltage_full': current_target_full,
            'error': day1_error,
            'day1_action': day1_action,
            'future_actions_plan': future_actions_denorm.copy(),
            'step': self.current_step,
            'cumulative_error': self.cumulative_error,
            'R_acc': reward_components['R_acc'],
            'P_smooth': reward_components['P_smooth'],
            'P_bound': reward_components['P_bound'],
            'P_unc': reward_components.get('P_unc', 0.0),
            'uncertainty_mean': float(np.mean(uncertainty)) if uncertainty is not None else 0.0,
        }

        return next_state, reward, done, info
    
    def get_state_dim(self):
        """获取状态维度"""
        return self.state_dim
    
    def get_action_dim(self):
        """获取动作维度"""
        return ACTION_TRAJECTORY_DIM


def load_predictor_model(model_path, num_pots, input_dim, device='cpu',
                          ensemble=False, ensemble_checkpoint_paths=None):
    """
    加载训练好的条件电压预测模型，自动处理新旧架构不兼容。

    参数:
        model_path: 单模型 .pth 路径（ensemble=False时必需）
        num_pots: 槽号数量
        input_dim: 输入特征维度
        device: 设备
        ensemble: True 表示加载 ensemble predictor
        ensemble_checkpoint_paths: ensemble 成员 .pth 路径列表
    返回:
        predictor: ConditionalVoltagePredictor 或 UncertaintyQuantifiedPredictor
    """
    if ensemble and ensemble_checkpoint_paths is not None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model'))
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from ensemble_predictor import UncertaintyQuantifiedPredictor
        predictor = UncertaintyQuantifiedPredictor(
            num_models=len(ensemble_checkpoint_paths),
            input_dim=input_dim, num_pots=num_pots, device=device
        )
        predictor.load_checkpoints(ensemble_checkpoint_paths)
        return predictor

    checkpoint = torch.load(model_path, map_location=device)

    # Detect actual num_pots from checkpoint (may differ if augmented data added pots)
    ckpt_num_pots = None
    pot_key = 'base_model.pot_embedding.weight'
    if pot_key in checkpoint:
        ckpt_num_pots = checkpoint[pot_key].shape[0]
    if ckpt_num_pots is not None and ckpt_num_pots != num_pots:
        print(f"   Note: checkpoint has {ckpt_num_pots} pots, RL config expects {num_pots}")
        actual_num_pots = ckpt_num_pots
    else:
        actual_num_pots = num_pots

    base_model = LSTMModelWithPotEmbedding(
        input_dim=input_dim,
        num_pots=actual_num_pots,
        pot_embed_dim=POT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_len=OUTPUT_LEN,
        dropout=DROPOUT
    ).to(device)

    predictor = ConditionalVoltagePredictor(
        base_model=base_model,
        future_action_dim=2,
        future_len=OUTPUT_LEN,
        cond_hidden=COND_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    # 加载模型权重（兼容新旧架构）
    try:
        predictor.load_state_dict(checkpoint)
        print(f"   条件预测模型加载成功: {os.path.basename(model_path)}")
    except RuntimeError as e:
        # 架构不兼容（旧88维特征/Conv1D → 新12维/LSTM）
        missing = [k for k in predictor.state_dict().keys()
                   if k not in checkpoint and 'num_batches_tracked' not in k]
        unexpected = [k for k in checkpoint.keys()
                      if k not in predictor.state_dict()]
        print(f"[WARN] 条件预测模型架构不兼容，无法加载")
        print(f"   缺失key: {missing[:4]}... (共{len(missing)}个)")
        print(f"   多余key: {unexpected[:4]}... (共{len(unexpected)}个)")
        print(f"   请重新训练条件预测器: cd model && python train.py (use_conditional=True)")
        raise RuntimeError(
            "条件预测模型与当前架构不兼容，需重新训练。\n"
            "步骤: 1) cd model && python train.py (use_conditional=False) 训练基础LSTM\n"
            "      2) cd model && python train.py (use_conditional=True)  训练条件预测器\n"
            "      3) 再运行本脚本"
        )

    predictor.eval()
    return predictor