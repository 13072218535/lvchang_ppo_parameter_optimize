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
OUTPUT_DIR = 'e:/TraeWorkplace/铝厂/2026-5-12-参数优化/model/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 序列长度配置
INPUT_LEN = 7       # 输入序列长度（前7天）
OUTPUT_LEN = 14     # 输出序列长度（预测后14天）

# 模型参数
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
POT_EMBED_DIM = 16

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
    '工作平均',          # 目标变量（用于历史序列输入）
]

# 目标变量
TARGET = '工作平均'
