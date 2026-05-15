# 基于PPO算法的参数优化改进方案

## 一、背景与目标

当前 `ppo参数优化/model/` 已实现完整的 MPD-PPO 流水线：条件电压预测器 → 仿真环境 → Actor-Critic网络 → PPO训练。但根据问题分析文档，存在两个 P0 级关键缺陷和若干 P1/P2 问题，导致优化方向错误和训练-部署不一致。

本方案在**保留PPO算法框架**的前提下，对上述问题进行系统性修复。

---

## 二、核心改进：从"单步交互"升级为"滚动时域轨迹PPO"

### 2.1 当前结构矛盾

```
当前:  PPO步进交互(逐天决策) ←→ 预测器批量预测(14天计划)
       ↓ 通过"保持策略"桥接 → 动作序列失真、误差累积、分布偏移
```

### 2.2 改进方案：Trajectory-Aware PPO

**核心思路**：让Actor一次性输出**完整的14天动作轨迹**（28维），与条件预测器的设计天然对齐。

```
改进后:
  Actor(s_t) → [a₁, a₂, ..., a₁₄] (28维动作轨迹)
  → 预测器(past_7d, 28维计划) → [v̂₁, ..., v̂₁₄]
  → 奖励 = Σ_{i=1}^{14} w_i · exp(-κ · (v̂_i - v_set_i)²)
  → 仅执行a₁更新状态 → 滑动窗口 → 下一步重新规划(滚动时域)
```

**优势**：
- 消除训练/推理分布不一致（预测器输入始终为Actor自洽生成的完整轨迹）
- 奖励覆盖全部14天预测，Agent必须考虑长期电压质量
- 从根本上避免预测电压回灌误差累积
- 保持PPO框架不变，仅扩展动作维度

---

## 三、具体修改内容

### 修改1：优化目标切换 — 从历史电压到设定电压（P0）

**文件**：`train_ppo.py` 的 `load_data_for_ppo()`

**当前**：
```python
target_voltage = pot_targets[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]
# pot_targets = pot_data['工作平均'].values  ← 历史实际电压
```

**修改为**：
```python
pot_set_voltages = pot_data['实际设定'].values  # 设定电压
target_voltage = pot_set_voltages[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]
```

**影响**：Actor的 `target_voltage` 输入在训练和部署时保持一致（均为设定电压），消除分布偏移。

---

### 修改2：Actor输出14天完整轨迹（P0 + P1）

**文件**：`ppo.py` 的 `Actor` 类

**当前**：输出2维单步动作 `[alf, out]`

**修改为**：输出28维14天轨迹 `[alf₁, out₁, alf₂, out₂, ..., alf₁₄, out₁₄]`

具体改动：
- `action_dim = 28`（原为2）
- 输出头改为 `mean_head: Linear(hidden//2, 28)`, `std_head: Linear(hidden//2, 28)`
- 每个维度独立 tanh 归一化到 [-1, 1]
- `sample_action()` 和 `get_log_prob()` 适配28维输出

**差异化裁剪保留**：
- ALF维度（索引 0,2,4,...,26）：`ε_clip = 0.1`
- OUT维度（索引 1,3,5,...,27）：`ε_clip = 0.2`
- 使用**独立裁剪+求和**替代乘积耦合

```
# 旧方案（乘积耦合）
clipped_ratio = clipped_alf_ratio * clipped_out_ratio  # 有效范围[0.72, 1.32]

# 新方案（独立裁剪+求和）
alf_surr = min(alf_ratio * A, clip(alf_ratio, 0.9, 1.1) * A)
out_surr = min(out_ratio * A, clip(out_ratio, 0.8, 1.2) * A)
surr = alf_surr + out_surr  # 两个头独立计算，梯度解耦
```

---

### 修改3：多步加权奖励函数（P1）

**文件**：`environment.py` 的 `_calculate_reward()`

**当前**：仅用 `voltage_pred[0]` 计算奖励

**修改为**：
```python
# 时间衰减权重：越近期的预测越重要
w_i = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

R_acc = Σ_{i=0}^{13} w_i * REWARD_ACC_WEIGHT * exp(-κ * (v̂_i - v_set_i)²)
```

权重总和归一化，保证与单步奖励可比。

---

### 修改4：环境交互重构 — 滚动时域（P0）

**文件**：`environment.py` 的 `step()` 方法

**当前流程**：
```
接收2维动作 → _build_future_actions(保持策略填14天) → 预测14天 → 取第1天算奖励 → 滑动窗口
```

**改进流程**：
```
接收28维动作轨迹 → 重塑为(14, 2) → 预测14天 → 取全部14天算多步奖励 → 取第1天动作更新状态 → 滑动窗口
```

关键变化：
- 移除 `_build_future_actions()`（不再需要保持策略填充）
- `step(action_28d)` 直接将完整轨迹输入预测器
- 状态更新只执行 day-1 动作（滚动时域机制）
- 14天预测全部参与奖励计算

---

### 修改5：非控制特征的状态转移模型（P1）

**文件**：`environment.py` 的 `_update_state()`

**当前**：仅更新 `工作平均`、`ALF加料量`、`实际出铝量` 三个特征，其余冻结

**改进**：增加经验更新规则

```python
# 铝水平：受出铝量直接影响（经验关系）
if '铝水平' in feature_cols:
    delta_out = action[1] - prev_out  # 出铝量变化
    al_level_change = -0.3 * (delta_out / prev_out) * al_level  # 经验系数
    features[-1, al_idx] += al_level_change

# 电解质水平：受ALF加料量影响
if '电解质水平' in feature_cols:
    delta_alf = action[0] - prev_alf
    electrolyte_change = 0.02 * delta_alf  # 经验系数
    features[-1, elec_idx] += electrolyte_change

# 平均电压：电压偏差传导
if '平均电压' in feature_cols:
    features[-1, avg_v_idx] = voltage_pred[0]  # 直接用预测值
```

使用可配置的经验系数，便于后续根据实际数据标定。

---

### 修改6：PPO更新逻辑修复（P2）

**文件**：`ppo.py` 的 `MPDPPO.update()`

1. **独立裁剪求和**（替代乘积耦合，见修改2）
2. **修正 batch_size 语义**：
   - 将 `PPO_BATCH_SIZE` 改为 `PPO_STEPS_PER_UPDATE = 256`（标准PPO步数）
   - 增加内层 mini-batch 切分：`mini_batch_size = 64`
3. **GAE 计算处理 episode 边界**：在 `compute_gae()` 中增加 `dones` 掩码的正确处理

---

## 四、训练流程改进

### `train_ppo.py` 修改点

1. **数据加载**：`load_data_for_ppo()` 中 `target_voltage` 改为 `实际设定`
2. **状态维度**：`state_dim = INPUT_LEN * input_dim + OUTPUT_LEN + 1`（不变）
3. **动作维度**：`action_dim = 28`（从2改为28）
4. **环境交互**：`run_episode()` 中每步收集28维动作
5. **新增训练配置**：
   ```python
   PPO_STEPS_PER_UPDATE = 256   # 每次更新前收集的步数
   PPO_MINI_BATCH_SIZE = 64     # 内层mini-batch大小
   PPO_INNER_EPOCHS = 10        # 每次更新内层迭代轮数
   REWARD_TIME_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
   ```

---

## 五、修改文件清单

| 文件 | 修改范围 | 关键变更 |
|------|----------|----------|
| `ppo参数优化/model/config.py` | 新增参数 | action_dim=28, PPO步数参数, 时间衰减权重, 经验系数 |
| `ppo参数优化/model/ppo.py` | 重构 | Actor输出28维, 独立裁剪求和, mini-batch更新 |
| `ppo参数优化/model/environment.py` | 重构 | step接收28维轨迹, 多步加权奖励, 非控特征更新 |
| `ppo参数优化/model/train_ppo.py` | 修改 | target_voltage=设定电压, 动作维度适配 |

---

## 六、滚动时域PPO vs 原方案对比

| 维度 | 原方案 | 改进方案 |
|------|--------|----------|
| 优化目标 | 匹配历史电压 | 逼近设定电压 |
| 动作空间 | 2维(单步) | 28维(14天轨迹) |
| 未来动作填充 | 保持策略(失真) | Actor自洽生成(一致) |
| 奖励计算 | 仅第1天 | 14天加权 |
| 裁剪方式 | 乘积耦合 | 独立求和 |
| 状态更新 | 3维更新 | 多特征经验更新 |
| 预测器输入 | 训练/推理不一致 | 完全一致 |

---

## 七、实施顺序

1. **第一轮**（修改1+4）：目标切换 + 环境交互重构 → 验证优化方向正确
2. **第二轮**（修改2+3+6）：Actor扩展28维 + 多步奖励 + 裁剪修复 → 完整PPO升级
3. **第三轮**（修改5）：非控特征更新规则 → 仿真物理合理性提升
4. **第四轮**：超参数调优（奖励权重、衰减系数、经验系数）

---

## 八、验证方案

1. **单元验证**：环境 `reset()` → `step()` 返回形状和数值正确性检查
2. **一致性验证**：对比Actor的策略输出与预测器输入，确认无分布偏移
3. **训练验证**：运行100轮PPO训练，观察奖励曲线是否收敛、Actor/Critic loss是否稳定
4. **效果验证**：在测试槽号上对比优化前后的预测电压与设定电压的MAE，预期MAE < 0.02V
5. **鲁棒性验证**：对不同槽号、不同初始状态进行推理，确认策略泛化能力
