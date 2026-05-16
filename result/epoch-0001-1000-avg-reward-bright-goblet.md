# PPO训练Reward异常分析与修复方案

## Context

PPO训练出现Reward极负(-2000~-3200)且持续恶化、Actor Loss≈0的问题。经代码分析发现**3个Critical级bug**和**2个High级设计缺陷**，导致训练信号完全消失。

## 问题诊断

### Bug-1 (Critical): `_update_state()`中target_voltage逐步缩减 → 形状崩溃
**文件**: `environment.py:297`
```python
self.target_voltage = self.target_voltage[1:]  # 14→13→12...
```
每步将target从14元素缩减到13。第2步`voltage_pred(14,)`与`target(13,)`形状不匹配。能跑通仅因`MAX_CUMULATIVE_ERROR=0.3V`在随机策略下第1步就触发done。

### Bug-2 (Critical): `compute_gae()`中reward裁剪[-50,50] 抹除全部学习信号
**文件**: `ppo.py:370`
```python
rewards = np.clip(rewards, -50, 50)
```
原始单步reward约-2000~-2300，全部被裁到-50。所有transition的reward完全相同，GAE退化为纯噪声，标准化后advantage≈0 → Actor Loss≈0。

### Bug-3 (Critical): 平滑惩罚(OUT)量级放大 -2260/step 碾压精度奖励
**文件**: `environment.py:139-157`
OUT动作范围[3500,4200] = 700kg跨度，`MAX_CHANGE=100kg`。随机策略相邻天expected diff ≈ 274kg → violation=174 → penalty=-174/pair × 13对 = **-2262/step**。对比R_acc max=+67，惩罚是奖励的33倍。

### Bug-4 (High): 初始std=0.694导致过度随机探索
**文件**: `ppo.py:110,145`
`gain=0.01`正交初始化 + softplus → std≈0.694。14天×2动作独立采样，相邻天差异巨大。

### Bug-5 (High): `SMOOTH_VIOLATION_WEIGHT=1.0`与原始violation量级不匹配
**文件**: `config.py:98`
violation单位是kg（可达几百），weight=1.0意味着1kg violation = 1.0 penalty，而R_acc仅+67。

---

## 修复方案（按优先级）

### 修复1: target_voltage滑动用填充替代缩减 (P0)
**文件**: `environment.py` `_update_state()` ~L297
```
# 旧: self.target_voltage = self.target_voltage[1:]
# 新: self.target_voltage = np.append(self.target_voltage[1:], self.target_voltage[-1])
```
保持14元素长度，用末值填充。

### 修复2: 平滑惩罚归一化到比例空间 (P0)
**文件**: `environment.py` `_calculate_reward()` ~L139-157
将violation除以对应动作范围，使ALF和OUT惩罚在同一量级:
```python
alf_range = self.alf_max - self.alf_min  # 15
out_range = self.out_max - self.out_min  # 700
alf_violation = max(0, alf_diff - self.alf_max_change) / alf_range
out_violation = max(0, out_diff - self.out_max_change) / out_range
```
调整weight为`REWARD_SMOOTH_VIOLATION_WEIGHT = 10.0`（与R_acc量级匹配）。

### 修复3: 移除reward硬裁剪，改用reward标准化 (P0)
**文件**: `ppo.py` `compute_gae()` ~L370
```
# 旧: rewards = np.clip(rewards, -50, 50)
# 新: 移除硬裁剪，在GAE循环前对rewards做标准化（0均值1方差）
```
与已修正的奖励量级配合，保留相对好坏信号。

### 修复4: 降低Actor初始std (P1)
**文件**: `ppo.py` Actor `__init__()` ~L110
将`std_head`的gain从0.01改为0.001，或添加可学习`log_std`参数（初始-1.0对应std≈0.37）。

### 修复5: 增加奖励分量日志 (P1)
**文件**: `train_ppo.py` 训练循环
在`run_episode()`中收集并打印R_acc/P_smooth/P_bound分量均值。

---

## 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `ppo参数优化/model/environment.py` | 修复1(target填充)、修复2(平滑惩罚归一化) |
| `ppo参数优化/model/ppo.py` | 修复3(reward标准化)、修复4(std初始化) |
| `ppo参数优化/model/config.py` | 修复2配套(SMOOTH_VIOLATION_WEIGHT改为10.0) |
| `ppo参数优化/model/train_ppo.py` | 修复5(分量日志) |

## 验证方案

1. 修改后`_calculate_reward()`单步返回应在[-30, +70]范围（而非-2300）
2. GAE不再全相同值（std > 0）
3. Critic Loss应能收敛（从~1.0下降）
4. Actor Loss应有非零值（策略在更新）
5. Avg Reward应在训练前期有上升趋势
