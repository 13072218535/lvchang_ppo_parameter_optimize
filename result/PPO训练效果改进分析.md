# PPO训练效果改进分析 — 第三轮深度诊断

## 背景

经过前两轮修复（奖励函数归一化、GAE修复、特征降维、条件编码器LSTM化），PPO训练的MAE仍然>1V。经对完整数据流的逐行追踪，发现根因是**标准化/反标准化不一致**——预测器训练时使用标准化数据，PPO仿真时传入原始物理量，导致量级错配。

---

## 根因：标准化-反标准化三重不一致

条件预测器在 `model/train.py` 中训练时，`DataProcessor.process()` 对所有特征（包括未来动作和目标变量）做了 `StandardScaler` 标准化，模型学到的映射是：

```
标准化past_features(μ≈0,σ≈1) + 标准化future_actions(μ≈0,σ≈1) → 标准化电压(μ≈0,σ≈1)
```

但在PPO环境 `environment.py` 中，三处出现**原始值替代标准化值**的错误：

### Bug 1 (Critical): future_actions 未标准化 → LSTM条件编码器饱和

**位置**: `environment.py:_predict_voltage()` L105

```python
# 现状：直接将原始物理量传入预测器
future_actions_tensor = torch.FloatTensor(future_actions).unsqueeze(0)  # ALF~32kg, OUT~3800kg
```

预测器的条件编码器是LSTM（input_size=2），训练时输入标准化值(μ≈0, σ≈1)，仿真时收到OUT=3800 → tanh(3800×w) 饱和为±1 → **所有动作信息丢失** → 预测器退化为不看动作的常量预测器。

### Bug 2 (Critical): 标准化预测值 vs 原始目标值 → 奖励恒为0

**位置**: `environment.py:_calculate_reward()` L133-134, `train_ppo.py:load_data_for_ppo()` L86

```python
# 预测器输出标准化电压 ~0.1
voltage_pred ≈ [0.10, 0.08, ...]   # 标准化单位

# 目标电压是原始设定值 ~4.1V
target_voltage = pot_data['实际设定'].values  # 原始V, ~4.08V

# 误差计算
errors = |0.10 - 4.08| ≈ 3.98V  → R_acc = 10 × exp(-2×3.98²) ≈ 0
```

每步误差~4V → cumulative_error一步超1.5V → episode立即终止 → PPO收不到任何学习信号。

### Bug 3 (Critical): 标准化电压写入原始特征 → 第二步状态彻底损坏

**位置**: `environment.py:_update_state()` L196, L219

```python
# voltage_pred[0] ≈ 0.10 (标准化值)
new_past_features[-1, target_idx] = np.clip(voltage_pred[0], 2.5, 6.0)
# → 工作平均 = 0.10V (实际应≈4.10V)

new_past_features[-1, avg_v_idx] = np.clip(voltage_pred[0], 3.0, 5.0)
# → 平均电压 = 3.0V (实际应≈4.09V)
```

第二步 scaler.transform 时：
- `(0.10 - 4.10) / 0.15 ≈ -26.7` → LSTM收到极端OOD值 → 输出垃圾

**数值追踪示例**（假设 scaler mean=4.10, std=0.15）：

| 步 | 工作平均(原始) | 标准化后 | 预测器输出(标准化) | 误差 | reward |
|----|--------------|---------|-----------------|------|--------|
| 1 | 4.10 (真实) | 0.00 | 0.10 | \|0.10-4.08\|=3.98V | ≈0 |
| 2 | 0.10 (被污染) | -26.7 | 随机值 | 极大 | 0 |

---

## 修复方案

### 修复1: 在 _predict_voltage 中标准化 future_actions

在传入预测器之前，用scaler的ALF/OUT列的均值和标准差对future_actions做标准化：

```python
# 获取scaler中ALF加料量(实际)和实际出铝量的mean/std
alf_mean = self.scaler.mean_[self.feature_indices['alf_actual']]
alf_scale = self.scaler.scale_[self.feature_indices['alf_actual']]
out_mean = self.scaler.mean_[self.feature_indices['out_actual']]
out_scale = self.scaler.scale_[self.feature_indices['out_actual']]

future_actions_std = future_actions.copy()
future_actions_std[:, 0] = (future_actions[:, 0] - alf_mean) / alf_scale
future_actions_std[:, 1] = (future_actions[:, 1] - out_mean) / out_scale
```

### 修复2: 将模型输出反标准化回原始电压

在 `_predict_voltage` 返回之前，将标准化电压反标准化为原始电压单位：

```python
target_mean = self.scaler.mean_[self.feature_indices['target']]  # 工作平均的mean
target_scale = self.scaler.scale_[self.feature_indices['target']]  # 工作平均的std
voltage_pred_raw = voltage_pred * target_scale + target_mean
```

### 修复3: 反标准化后再写入特征 + 计算奖励

修复2之后，`voltage_pred` 变为原始V单位(~4.1V)，与 `target_voltage` 一致，误差计算和特征更新均正常。

---

## 修复影响

| 修复前 | 修复后 |
|--------|--------|
| future_actions 原始值 → LSTM饱和 | 标准化值 → LSTM正常工作 |
| 误差 ≈ 4V → reward ≈ 0 | 误差 ≈ 0.1V → reward ≈ 40-60 |
| episode 1步终止 | episode 可走满5步 |
| 状态特征被污染 | 状态特征保持一致 |

---

## 完整修复清单（三阶段累计）

### 已修复（第一、二轮）
- 平滑惩罚归一化 + 权重调整
- reward标准化替代硬裁剪
- Critic值域收紧[-50,50]
- Actor初始std降低
- 特征维度88→12
- 条件编码器Conv1D→LSTM
- MAX_EPISODE_STEPS 14→5
- MAX_CUMULATIVE_ERROR 0.3→1.5
- target_voltage填充替代缩减
- 预测器预训练路径修复

### 本轮修复（第三轮）
| # | 修复项 | 文件 | 改动 |
|---|--------|------|------|
| 13 | future_actions标准化 | `environment.py:_predict_voltage()` | ~8行 |
| 14 | 预测电压反标准化 | `environment.py:_predict_voltage()` | ~3行 |
| 15 | 移除_update_state中重复钳制 | `environment.py:_update_state()` | 简化 |

### 预期效果
修复后，预测器在PPO环境中的预测值回到物理合理范围（~4V），episode能完整运行，reward有区分度。预期MAE从 >1V 降至与预测器自身精度（~0.1V）相当的水平。

---

## 验证方案

1. 在 `_predict_voltage` 返回前打印 `voltage_pred_raw[:3]`，确认值在~4V范围
2. 检查环境 `step()` 的 info 中 `voltage_pred` 和 `target_voltage_set` 值是否在同一量级
3. 运行PPO训练，观察reward是否从~0变为正值范围[20, 60]
4. 观察Critic Loss是否开始下降（从~1.0开始收敛）
5. 观察Actor Loss是否出现非零值（策略在学习）
