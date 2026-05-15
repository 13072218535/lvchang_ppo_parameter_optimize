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
        self.best_error = float('inf')
        self.initial_error = None
        
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
    
    def _apply_smooth_constraint(self, action):
        """应用动作平滑约束（软约束，返回平滑惩罚值）"""
        smooth_penalty = 0.0
        if self.last_action is not None:
            alf_diff = action[0] - self.last_action[0]
            out_diff = action[1] - self.last_action[1]
            
            alf_violation = max(0, abs(alf_diff) - self.alf_max_change)
            out_violation = max(0, abs(out_diff) - self.out_max_change)
            
            if alf_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * alf_violation
            if out_violation > 0:
                smooth_penalty -= REWARD_SMOOTH_VIOLATION_WEIGHT * out_violation
        
        return smooth_penalty
    
    def _build_future_actions(self, current_action):
        """
        构建未来14天的动作序列
        使用保持策略：当前动作 + 后续使用历史平均值
        """
        # 创建未来14天动作序列
        future_actions = np.zeros((OUTPUT_LEN, 2))
        
        # 第1天使用当前动作
        future_actions[0] = current_action
        
        # 第2到14天使用保持策略（重复前一天的动作）
        for i in range(1, OUTPUT_LEN):
            future_actions[i] = current_action  # 保持策略
        
        return future_actions
    
    def _predict_voltage(self, past_features, future_actions, pot_id):
        """调用条件预测模型预测电压"""
        self.predictor.eval()
        with torch.no_grad():
            past_tensor = torch.FloatTensor(past_features).unsqueeze(0).to(self.device)
            future_actions_tensor = torch.FloatTensor(future_actions).unsqueeze(0).to(self.device)
            pot_id_tensor = torch.LongTensor([pot_id]).to(self.device)
            
            voltage_pred = self.predictor(past_tensor, future_actions_tensor, pot_id_tensor)
            return voltage_pred.cpu().numpy()[0]
    
    def _calculate_reward(self, voltage_pred, target_voltage, action):
        """
        计算奖励函数
        R_t = R_acc + R_prog + P_smooth + P_bound
        """
        e_t = abs(voltage_pred[0] - target_voltage[0])

        R_acc = REWARD_ACC_WEIGHT * np.exp(-REWARD_ACC_KAPPA * (e_t ** 2))

        if self.best_error == float('inf'):
            improvement = 0.0
        else:
            improvement = self.best_error - e_t
        self.best_error = min(self.best_error, e_t)
        R_prog = REWARD_PROG_WEIGHT * improvement

        smooth_penalty = self._apply_smooth_constraint(action)

        P_bound = 0.0
        alf_range = self.alf_max - self.alf_min
        out_range = self.out_max - self.out_min
        if action[0] <= self.alf_min:
            P_bound -= REWARD_BOUND_PENALTY * ((self.alf_min - action[0]) / alf_range)
        elif action[0] >= self.alf_max:
            P_bound -= REWARD_BOUND_PENALTY * ((action[0] - self.alf_max) / alf_range)
        if action[1] <= self.out_min:
            P_bound -= REWARD_BOUND_PENALTY * ((self.out_min - action[1]) / out_range)
        elif action[1] >= self.out_max:
            P_bound -= REWARD_BOUND_PENALTY * ((action[1] - self.out_max) / out_range)

        reward = R_acc + R_prog + smooth_penalty + P_bound

        if np.isinf(reward) or np.isnan(reward):
            reward = R_acc
        
        if reward == 0.0 or np.isclose(reward, 0.0, atol=1e-10):
            print(f"DEBUG: reward=0, e_t={e_t}, R_acc={R_acc}, R_prog={R_prog}, smooth_penalty={smooth_penalty}, P_bound={P_bound}")
            print(f"  voltage_pred={voltage_pred}, target_voltage={target_voltage}")

        return reward, e_t
    
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
    
    def _update_state(self, action, voltage_pred):
        """
        更新状态：滑动时间窗口，构造新的过去7天状态
        参数:
            action: [alf, out] 当前动作
            voltage_pred: 预测的电压序列
        """
        # 复制当前过去特征
        new_past_features = self.past_features.copy()
        
        # 滑动窗口：丢弃最老的一天，加入新的一天（复制倒数第二天）
        new_past_features = np.concatenate([new_past_features[1:], new_past_features[-1:]], axis=0)
        
        # 获取最新一天的数据（需要更新的行）
        last_day_idx = -1
        
        # 更新目标变量（工作平均电压）为预测值
        if self.feature_indices['target'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['target']] = voltage_pred[0]
        
        # 更新ALF加料量（实际）为当前动作
        if self.feature_indices['alf_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['alf_actual']] = action[0]
        
        # 更新实际出铝量为当前动作
        if self.feature_indices['out_actual'] >= 0:
            new_past_features[last_day_idx, self.feature_indices['out_actual']] = action[1]
        
        # 更新统计特征（基于新的窗口重新计算）
        self._update_statistical_features(new_past_features)
        
        # 更新累计误差
        self.cumulative_error += abs(voltage_pred[0] - self.target_voltage[0])
        
        # 更新目标电压序列（滑动一天）
        self.target_voltage = self.target_voltage[1:]
        
        return new_past_features
    
    def reset(self, past_features, target_voltage, pot_id):
        """
        重置环境到初始状态
        参数:
            past_features: (7, feature_dim) 过去7天的特征
            target_voltage: (14,) 未来14天的目标电压序列
            pot_id: 槽号索引
        """
        self.past_features = past_features.copy()
        self.target_voltage = target_voltage.copy()
        self.pot_id = pot_id
        self.current_step = 0
        self.last_action = None
        self.cumulative_error = 0.0
        self.best_error = float('inf')
        
        # 计算状态维度
        self.state_dim = past_features.shape[1] + len(target_voltage) + 1
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态向量 - 包含完整7天历史窗口"""
        past_features_flat = self.past_features.flatten()
        state = np.concatenate([past_features_flat, self.target_voltage, [self.pot_id]])
        return state
    
    def step(self, action):
        """
        执行一步动作
        参数:
            action: [alf, out] 归一化到[-1,1]范围的动作
        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否终止
            info: 附加信息
        """
        action = self._normalize_action(action)
        
        action = self._clip_action(action)
        
        future_actions = self._build_future_actions(action)
        
        voltage_pred = self._predict_voltage(self.past_features, future_actions, self.pot_id)
        
        reward, error = self._calculate_reward(voltage_pred, self.target_voltage, action)
        
        self.past_features = self._update_state(action, voltage_pred)
        
        self.current_step += 1
        
        done = False
        if self.current_step >= MAX_EPISODE_STEPS:
            done = True
        if self.cumulative_error >= MAX_CUMULATIVE_ERROR:
            done = True
        
        self.last_action = action.copy()
        
        next_state = self._get_state()
        
        info = {
            'voltage_pred': voltage_pred[0],
            'target_voltage': self.target_voltage[0] if len(self.target_voltage) > 0 else None,
            'error': error,
            'action': action,
            'step': self.current_step,
            'cumulative_error': self.cumulative_error
        }
        
        return next_state, reward, done, info
    
    def get_state_dim(self):
        """获取状态维度"""
        return self.state_dim
    
    def get_action_dim(self):
        """获取动作维度"""
        return 2


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