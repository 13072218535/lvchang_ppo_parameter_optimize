"""
MPD-PPO仿真环境类
封装条件电压预测模型，实现MDP转移
"""
import numpy as np
import torch
import torch.nn as nn

from model import LSTMModelWithPotEmbedding, ConditionalVoltagePredictor
from config import *


class VoltageControlEnv:
    """电压控制仿真环境"""
    
    def __init__(self, predictor_model, scaler, feature_cols, device='cpu'):
        """
        参数:
            predictor_model: 训练好的条件电压预测模型
            scaler: 数据标准化器
            feature_cols: 特征列名列表（用于索引映射）
            device: 设备（cpu或cuda）
        """
        self.predictor = predictor_model
        self.scaler = scaler
        self.device = device
        self.feature_cols = feature_cols
        
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

        self._actor_hidden = None
    
    def _build_feature_indices(self):
        """构建特征列索引映射"""
        self.feature_indices = {
            # 原始特征索引
            'target': self.feature_cols.index('工作平均') if '工作平均' in self.feature_cols else -1,
            'alf_actual': self.feature_cols.index('ALF加料量(实际)') if 'ALF加料量(实际)' in self.feature_cols else -1,
            'out_actual': self.feature_cols.index('实际出铝量') if '实际出铝量' in self.feature_cols else -1,
            'alf_set': self.feature_cols.index('ALF加料量(设定)') if 'ALF加料量(设定)' in self.feature_cols else -1,
            # 目标变量统计特征索引
            'target_mean_3d': self.feature_cols.index('工作平均_mean_3d') if '工作平均_mean_3d' in self.feature_cols else -1,
            'target_std_3d': self.feature_cols.index('工作平均_std_3d') if '工作平均_std_3d' in self.feature_cols else -1,
            'target_mean_7d': self.feature_cols.index('工作平均_mean_7d') if '工作平均_mean_7d' in self.feature_cols else -1,
            'target_std_7d': self.feature_cols.index('工作平均_std_7d') if '工作平均_std_7d' in self.feature_cols else -1,
            'target_diff_1': self.feature_cols.index('工作平均_diff_1') if '工作平均_diff_1' in self.feature_cols else -1,
            'target_diff_7': self.feature_cols.index('工作平均_diff_7') if '工作平均_diff_7' in self.feature_cols else -1,
            # ALF实际统计特征索引
            'alf_mean_3d': self.feature_cols.index('ALF加料量(实际)_mean_3d') if 'ALF加料量(实际)_mean_3d' in self.feature_cols else -1,
            'alf_std_3d': self.feature_cols.index('ALF加料量(实际)_std_3d') if 'ALF加料量(实际)_std_3d' in self.feature_cols else -1,
            # 出铝量统计特征索引
            'out_mean_3d': self.feature_cols.index('实际出铝量_mean_3d') if '实际出铝量_mean_3d' in self.feature_cols else -1,
            'out_std_3d': self.feature_cols.index('实际出铝量_std_3d') if '实际出铝量_std_3d' in self.feature_cols else -1,
            # 衍生特征索引
            'voltage_diff': self.feature_cols.index('电压偏差') if '电压偏差' in self.feature_cols else -1,
        }
    
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
        """调用条件预测模型预测电压"""
        self.predictor.eval()
        with torch.no_grad():
            standardized_features = self.scaler.transform(past_features)
            past_tensor = torch.FloatTensor(standardized_features).unsqueeze(0).to(self.device)
            future_actions_tensor = torch.FloatTensor(future_actions).unsqueeze(0).to(self.device)
            pot_id_tensor = torch.LongTensor([pot_id]).to(self.device)
            
            voltage_pred = self.predictor(past_tensor, future_actions_tensor, pot_id_tensor)
            voltage_pred = voltage_pred.cpu().numpy()[0]
            # 钳制到物理合理范围，阻断正反馈回路导致电压指数爆炸
            voltage_pred = np.clip(voltage_pred, 2.5, 6.0)
            return voltage_pred
    
    def _calculate_reward(self, voltage_pred, target_voltage, action_trajectory):
        """
        计算多步加权奖励函数
        R_acc = Σ_{i=0}^{13} w_i * ACC_WEIGHT * exp(-κ * (v̂_i - v_set_i)²)
        R_t = R_acc + P_smooth + P_bound
        参数:
            voltage_pred: (14,) 预测电压序列
            target_voltage: (14,) 目标电压序列（设定电压）
            action_trajectory: (14, 2) 14天动作轨迹（未归一化）
        """
        # 多步加权精度奖励
        errors = np.abs(voltage_pred - target_voltage)
        time_weights = np.array(REWARD_TIME_WEIGHTS[:len(voltage_pred)])
        R_acc = REWARD_ACC_WEIGHT * np.sum(
            time_weights * np.exp(-REWARD_ACC_KAPPA * (errors ** 2))
        )

        # 平滑惩罚（检查相邻天动作变化，归一化到比例空间）
        alf_range = self.alf_max - self.alf_min
        out_range = self.out_max - self.out_min
        smooth_penalty = 0.0
        for i in range(1, len(action_trajectory)):
            alf_diff = abs(action_trajectory[i, 0] - action_trajectory[i-1, 0])
            out_diff = abs(action_trajectory[i, 1] - action_trajectory[i-1, 1])
            alf_violation = max(0, alf_diff - self.alf_max_change) / alf_range
            out_violation = max(0, out_diff - self.out_max_change) / out_range
            if alf_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * alf_violation
            if out_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * out_violation
        # 加上与上一步最后动作的平滑约束（如果存在）
        if self.last_action is not None:
            alf_diff = abs(action_trajectory[0, 0] - self.last_action[0])
            out_diff = abs(action_trajectory[0, 1] - self.last_action[1])
            alf_violation = max(0, alf_diff - self.alf_max_change) / alf_range
            out_violation = max(0, out_diff - self.out_max_change) / out_range
            if alf_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * alf_violation
            if out_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * out_violation

        # 边界惩罚
        P_bound = 0.0
        alf_range = self.alf_max - self.alf_min
        out_range = self.out_max - self.out_min
        for i in range(len(action_trajectory)):
            a = action_trajectory[i]
            if a[0] <= self.alf_min:
                P_bound -= REWARD_BOUND_PENALTY * ((self.alf_min - a[0]) / alf_range)
            elif a[0] >= self.alf_max:
                P_bound -= REWARD_BOUND_PENALTY * ((a[0] - self.alf_max) / alf_range)
            if a[1] <= self.out_min:
                P_bound -= REWARD_BOUND_PENALTY * ((self.out_min - a[1]) / out_range)
            elif a[1] >= self.out_max:
                P_bound -= REWARD_BOUND_PENALTY * ((a[1] - self.out_max) / out_range)

        reward = R_acc + smooth_penalty + P_bound

        if np.isinf(reward) or np.isnan(reward):
            reward = R_acc

        reward_components = {'R_acc': R_acc, 'P_smooth': smooth_penalty, 'P_bound': P_bound}
        return reward, errors[0], reward_components
    
    def _update_statistical_features(self, features):
        """
        更新统计特征（基于新的滑动窗口）
        参数:
            features: (7, feature_dim) 过去7天的特征
        """
        last_day_idx = -1
        
        # 更新目标变量的统计特征
        if self.feature_indices['target'] >= 0:
            target_vals = features[:, self.feature_indices['target']]
            
            # 3天均值/标准差（取最后3天）
            if self.feature_indices['target_mean_3d'] >= 0:
                features[last_day_idx, self.feature_indices['target_mean_3d']] = np.mean(target_vals[-3:])
            if self.feature_indices['target_std_3d'] >= 0:
                features[last_day_idx, self.feature_indices['target_std_3d']] = np.std(target_vals[-3:])
            
            # 7天均值/标准差（取全部7天）
            if self.feature_indices['target_mean_7d'] >= 0:
                features[last_day_idx, self.feature_indices['target_mean_7d']] = np.mean(target_vals)
            if self.feature_indices['target_std_7d'] >= 0:
                features[last_day_idx, self.feature_indices['target_std_7d']] = np.std(target_vals)
            
            # 1阶差分（与前一天的差值）
            if self.feature_indices['target_diff_1'] >= 0 and len(target_vals) >= 2:
                features[last_day_idx, self.feature_indices['target_diff_1']] = target_vals[-1] - target_vals[-2]
            
            # 7阶差分（与7天前的差值）
            if self.feature_indices['target_diff_7'] >= 0 and len(target_vals) >= 7:
                features[last_day_idx, self.feature_indices['target_diff_7']] = target_vals[-1] - target_vals[0]
        
        # 更新ALF加料量的统计特征
        if self.feature_indices['alf_actual'] >= 0:
            alf_vals = features[:, self.feature_indices['alf_actual']]
            if self.feature_indices['alf_mean_3d'] >= 0:
                features[last_day_idx, self.feature_indices['alf_mean_3d']] = np.mean(alf_vals[-3:])
            if self.feature_indices['alf_std_3d'] >= 0:
                features[last_day_idx, self.feature_indices['alf_std_3d']] = np.std(alf_vals[-3:])
        
        # 更新实际出铝量的统计特征
        if self.feature_indices['out_actual'] >= 0:
            out_vals = features[:, self.feature_indices['out_actual']]
            if self.feature_indices['out_mean_3d'] >= 0:
                features[last_day_idx, self.feature_indices['out_mean_3d']] = np.mean(out_vals[-3:])
            if self.feature_indices['out_std_3d'] >= 0:
                features[last_day_idx, self.feature_indices['out_std_3d']] = np.std(out_vals[-3:])
        
        # 更新电压偏差（衍生特征）
        if self.feature_indices['voltage_diff'] >= 0 and self.feature_indices['target'] >= 0:
            # 假设电压设定也在特征中
            if '电压设定' in self.feature_cols:
                voltage_set_idx = self.feature_cols.index('电压设定')
                features[last_day_idx, self.feature_indices['voltage_diff']] = \
                    features[last_day_idx, self.feature_indices['target']] - features[last_day_idx, voltage_set_idx]
    
    def _update_state(self, action, voltage_pred, prev_features):
        """
        更新状态：滑动时间窗口，构造新的过去7天状态
        包含非控制特征的经验更新规则
        参数:
            action: [alf, out] 当前执行的动作（未归一化）
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

        # 更新控制变量
        if self.feature_indices['alf_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['alf_actual']] = action[0]
        if self.feature_indices['out_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['out_actual']] = action[1]

        # === 非控制特征经验更新 ===
        prev_alf = prev_features[-1, self.feature_indices['alf_actual']] if self.feature_indices['alf_actual'] >= 0 else action[0]
        prev_out = prev_features[-1, self.feature_indices['out_actual']] if self.feature_indices['out_actual'] >= 0 else action[1]
        delta_alf = action[0] - prev_alf
        delta_out = action[1] - prev_out

        # 铝水平：出铝量增加 → 铝水平降低（经验关系）
        al_idx = self.feature_cols.index('铝水平') if '铝水平' in self.feature_cols else -1
        if al_idx >= 0 and prev_out > 0:
            al_change = EMPIRICAL_AL_OUT_RATIO * (delta_out / prev_out) * new_past_features[second_last_idx, al_idx]
            new_past_features[last_day_idx, al_idx] = new_past_features[second_last_idx, al_idx] + al_change

        # 平均电压：直接用预测电压更新，钳制防止正反馈爆炸
        avg_v_idx = self.feature_cols.index('平均电压') if '平均电压' in self.feature_cols else -1
        if avg_v_idx >= 0:
            new_past_features[last_day_idx, avg_v_idx] = np.clip(voltage_pred[0], 2.5, 6.0)

        # 电压设定：通常不变或缓慢变化，沿用最新值
        set_v_idx = self.feature_cols.index('电压设定') if '电压设定' in self.feature_cols else -1
        if set_v_idx >= 0:
            new_past_features[last_day_idx, set_v_idx] = new_past_features[second_last_idx, set_v_idx]

        # 实际设定：同电压设定
        actual_set_idx = self.feature_cols.index('实际设定') if '实际设定' in self.feature_cols else -1
        if actual_set_idx >= 0:
            new_past_features[last_day_idx, actual_set_idx] = new_past_features[second_last_idx, actual_set_idx]

        # 更新统计特征（基于新的窗口重新计算）
        self._update_statistical_features(new_past_features)

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
        """获取当前状态向量 - 包含完整7天历史窗口"""
        past_features_flat = self.past_features.flatten()
        state = np.concatenate([past_features_flat, self.target_voltage, [self.pot_id]])
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

        voltage_pred = self._predict_voltage(self.past_features, future_actions_denorm, self.pot_id)

        reward, day1_error, reward_components = self._calculate_reward(voltage_pred, self.target_voltage, future_actions_denorm)

        # 在滑动前保存当前天的目标电压（用于info输出）
        current_target = self.target_voltage[0] if len(self.target_voltage) > 0 else None
        current_target_full = self.target_voltage.copy()

        prev_features = self.past_features.copy()
        day1_action = future_actions_denorm[0]
        self.past_features = self._update_state(day1_action, voltage_pred, prev_features)

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
        }

        return next_state, reward, done, info
    
    def get_state_dim(self):
        """获取状态维度"""
        return self.state_dim
    
    def get_action_dim(self):
        """获取动作维度"""
        return ACTION_TRAJECTORY_DIM


def load_predictor_model(model_path, num_pots, input_dim, device='cpu'):
    """
    加载训练好的条件电压预测模型
    """
    # 创建基础LSTM模型
    base_model = LSTMModelWithPotEmbedding(
        input_dim=input_dim,
        num_pots=num_pots,
        pot_embed_dim=POT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_len=OUTPUT_LEN,
        dropout=DROPOUT
    ).to(device)
    
    # 创建条件预测器
    predictor = ConditionalVoltagePredictor(
        base_model=base_model,
        future_action_dim=2,
        future_len=OUTPUT_LEN,
        cond_hidden=COND_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    
    # 加载模型权重
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    predictor.eval()
    
    return predictor