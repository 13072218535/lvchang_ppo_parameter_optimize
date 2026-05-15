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
DATA_PATH = 'e:/TraeWorkplace/铝厂/2026-5-12-参数优化/槽况数据_处理后_v2.xlsx'
OUTPUT_DIR = 'e:/TraeWorkplace/铝厂/2026-5-12-参数优化/ppo参数优化/model/output'
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
PPO_EPOCHS = 1000              # PPO迭代轮数
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
REWARD_ACC_WEIGHT = 10.0       # 精度奖励权重
REWARD_ACC_KAPPA = 2.0         # 精度奖励指数衰减系数
REWARD_PROG_WEIGHT = 0.3       # 进度奖励权重
REWARD_SMOOTH_VIOLATION_WEIGHT = 1.0  # 平滑约束违反惩罚权重
REWARD_BOUND_PENALTY = 10      # 边界惩罚值（软约束）

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
MAX_EPISODE_STEPS = 14         # 最大步数（预测窗口）
MAX_CUMULATIVE_ERROR = 0.3     # 累计误差上限 (V)

# GAE参数
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95

# 保存间隔
SAVE_INTERVAL = 50             # 每50轮保存一次模型

# 日志间隔
LOG_INTERVAL = 10              # 每10轮打印一次日志