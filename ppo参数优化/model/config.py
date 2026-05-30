import os
import random
import numpy as np
import torch

# 随机种子（保证实验可复现）
SEED = 42

def set_seed(seed=SEED):
    """设置随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 数据路径
DATA_PATH = r'E:\ClaudeCodeWorkplace\2026-5-12-参数优化\槽况数据_处理后_v2.xlsx'
OUTPUT_DIR = r'E:\ClaudeCodeWorkplace\2026-5-12-参数优化\ppo参数优化\model\output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 序列长度配置
INPUT_LEN = 7       # 输入序列长度（前7天）
OUTPUT_LEN = 14     # 输出序列长度（预测后14天）

# 模型参数
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
POT_EMBED_DIM = 16

# 条件预测模型参数
COND_HIDDEN = 32

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# 早停与学习率调度
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# 槽号划分（用于槽号分割方式）
TRAIN_POTS = [1101, 1102, 1104, 1106, 1107, 1108, 1109, 1111, 1112, 1113,
              1114, 1115, 1116, 1117, 1118, 1119, 1121, 1122, 1123, 1124,
              1125, 1126, 1127, 1129, 1130, 1131, 1133, 1135, 1138, 1141]
VAL_POTS = [1103, 1120, 1128, 1132, 1136, 1140]
TEST_POTS = [1105, 1110, 1134, 1137, 1139, 1142]

# 工作电压预测高相关性特征（基于Spearman筛选，|r| >= 0.2，包含目标变量）
HIGH_CORR_FEATURES = [
    '平均电压',          # 极强相关 (0.9689)
    '实际设定',          # 强相关 (0.8829)
    '电压设定',          # 强相关 (0.7656)
    '铝水平',           # 中等相关 (0.6121)
    'ALF加料量(设定)',   # 中等相关 (-0.4265)
    '电解质水平',        # 中等相关 (-0.4253)
    'Fe含量(%)',        # 中等相关 (0.4080)
    '槽龄',             # 弱相关 (0.3789)
    'ALF加料量(实际)',   # 弱相关 (-0.2585)
    '电流效率(%)',       # 弱相关 (0.2428)
    '实际出铝量',        # 弱正相关 (0.18640)
    '工作平均'          # 目标变量
]

# 目标变量
TARGET = '工作平均'

# ==================== PPO相关参数 ====================

# PPO训练参数
PPO_EPOCHS = 200              # PPO迭代轮数
PPO_STEPS_PER_UPDATE = 256     # 每次更新前收集的步数（替代旧的PPO_BATCH_SIZE）
PPO_MINI_BATCH_SIZE = 64       # 内层mini-batch大小
PPO_INNER_EPOCHS = 10          # 每次更新的内层迭代轮数
PPO_LEARNING_RATE = 3e-4       # PPO学习率
PPO_GAMMA = 0.99               # 折扣因子
PPO_LAMBDA = 0.95              # GAE参数

# 差异化裁剪阈值（论文建议）
EPS_CLIP_ALF = 0.1             # ALF加料量裁剪范围 [0.9, 1.1]
EPS_CLIP_OUT = 0.2             # 实际出铝量裁剪范围 [0.8, 1.2]

# 动作维度：14天×2动作=28维（完整轨迹）
ACTION_TRAJECTORY_DIM = 28     # 14天完整动作轨迹维度

# 奖励函数权重
REWARD_ACC_WEIGHT = 8.0        # 精度奖励权重（配合κ=15，拉开好坏动作差距）
REWARD_ACC_KAPPA = 15.0        # 精度奖励指数衰减系数（原2.0过平，15.0使0.1V vs 0.3V差距达3.2x）
REWARD_PROG_WEIGHT = 0.3       # 进度奖励权重
REWARD_SMOOTH_VIOLATION_WEIGHT = 20.0  # 平滑约束违反惩罚权重（violation已归一化到[0,1]比例空间）
REWARD_SMOOTH_EXTREME_MULTIPLIER = 3.0  # 极端违反惩罚乘数：|Δ| > 2×上限时额外×3
REWARD_BOUND_PENALTY = 10      # 边界惩罚值（软约束）
REWARD_BOUND_MARGIN = 0.10     # 边界余量比例：动作进入[min, min+10%]或[max-10%, max]时触发惩罚

# ==================== MPD-PPO改进参数 ====================
# 改进1：轨迹平滑正则化（直接约束Actor输出的日间变化）
SMOOTH_REG_WEIGHT = 0.05       # 轨迹平滑正则权重（保守起步，避免与reward信号冲突）
SMOOTH_REG_ALF_THRESHOLD = 0.10  # ALF归一化日变化阈值（tanh空间≈1.5kg）
SMOOTH_REG_OUT_THRESHOLD = 0.07  # OUT归一化日变化阈值（tanh空间≈49kg）

# 改进3：熵退火调度（极小幅熵奖励，仅防策略过早坍缩）
ENTROPY_COEF_START = 0.002     # 初始熵系数（非常小）
ENTROPY_COEF_END = 0.0001      # 最终熵系数（几乎为零）
ENTROPY_DECAY_EPOCHS = 400     # 熵系数线性衰减到END的epoch数

# 多步奖励时间衰减权重（14天，逐日衰减）
REWARD_TIME_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4,
                       0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

# 动作范围（由历史数据5%~95%分位数确定）
ACTION_ALF_MIN = 25.0          # ALF加料量最小值 (kg)
ACTION_ALF_MAX = 40.0          # ALF加料量最大值 (kg)
ACTION_OUT_MIN = 3500.0        # 实际出铝量最小值 (kg)
ACTION_OUT_MAX = 4200.0        # 实际出铝量最大值 (kg)

# 动作变化约束
ACTION_ALF_MAX_CHANGE = 3.0    # ALF加料量日变化上限 (kg) 
ACTION_OUT_MAX_CHANGE = 100.0  # 实际出铝量日变化上限 (kg)

# 非控制特征经验更新系数
EMPIRICAL_AL_OUT_RATIO = -0.3  # 铝水平/出铝量经验系数
EMPIRICAL_ELECTROLYTE_ALF = 0.02  # 电解质水平/ALF加料量经验系数

# 终止条件
MAX_EPISODE_STEPS = 14          # 最大步数（标准化bug已修复，恢复完整14步轨迹）
MAX_CUMULATIVE_ERROR = 1.5      # 累计误差上限（放宽以允许agent体验完整轨迹，原0.3→1.5）

# GAE参数
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95

# 保存间隔
SAVE_INTERVAL = 50             # 每50轮保存一次模型

# ==================== TAA-PPO改进参数 ====================
# 改进1：时间自适应裁剪 — 14天×2动作的独立ε调度（平衡探索与稳定）
TAA_EPS_SCHEDULE = [
    # day 0-2 (紧): 实际执行的动作
    [0.10, 0.10], [0.10, 0.10], [0.10, 0.10],
    # day 3-6 (中): 近端计划
    [0.15, 0.15], [0.15, 0.15], [0.15, 0.15], [0.15, 0.15],
    # day 7-10 (中松): 中期计划
    [0.20, 0.20], [0.20, 0.20], [0.20, 0.20], [0.20, 0.20],
    # day 11-13 (松): 远期意图
    [0.25, 0.25], [0.25, 0.25], [0.25, 0.25],
]

# 改进2：裁剪率预热 — 温和预热以平衡速度和稳定
TAA_CLIP_WARMUP_EPOCHS = 80
TAA_CLIP_WARMUP_FACTOR = 1.8      # epoch0: ε×1.8, 线性退火到目标ε

# 改进3：自适应平滑正则 — 前期低权重不阻碍精度学习
TAA_SMOOTH_START_WEIGHT = 0.005   # 初始平滑权重（epoch < 30）
TAA_SMOOTH_END_WEIGHT = 0.05      # 最终平滑权重（epoch >= 100）
TAA_SMOOTH_RAMP_START = 30        # 权重增长起始epoch
TAA_SMOOTH_RAMP_END = 100         # 权重增长结束epoch

# 改进4：双Critic集成
TAA_USE_DUAL_CRITIC = True        # 是否使用双V-Critic（min聚合GAE）

# 改进5：LayerNorm替代BatchNorm（在Actor/Critic中）
TAA_USE_LAYER_NORM = True         # True=LayerNorm, False=BatchNorm1d

# 日志间隔
LOG_INTERVAL = 10              # 每10轮打印一次日志