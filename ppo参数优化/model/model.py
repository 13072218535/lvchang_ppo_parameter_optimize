import torch
import torch.nn as nn
from config import *


class LSTMModel(nn.Module):
    """基础LSTM模型（无槽嵌入）"""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, 
                 output_len=OUTPUT_LEN, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_len)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后时刻的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # BatchNorm
        last_hidden = self.bn(last_hidden)
        
        # 全连接层
        output = self.fc(last_hidden)  # (batch_size, output_len)
        
        return output


class LSTMModelWithPotEmbedding(nn.Module):
    """带槽嵌入的LSTM模型"""
    def __init__(self, input_dim, num_pots, pot_embed_dim=POT_EMBED_DIM, 
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, 
                 output_len=OUTPUT_LEN, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len
        
        # 槽嵌入层
        self.pot_embedding = nn.Embedding(num_pots, pot_embed_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim + pot_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_len)
        )
    
    def forward(self, x, pot_ids):
        # x: (batch_size, seq_len, input_dim)
        # pot_ids: (batch_size,)
        
        batch_size, seq_len, _ = x.shape
        
        # 获取槽嵌入并扩展为序列长度
        pot_embed = self.pot_embedding(pot_ids)  # (batch_size, pot_embed_dim)
        pot_embed = pot_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, pot_embed_dim)
        
        # 拼接特征和槽嵌入
        x = torch.cat([x, pot_embed], dim=-1)  # (batch_size, seq_len, input_dim + pot_embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后时刻的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # BatchNorm
        last_hidden = self.bn(last_hidden)
        
        # 全连接层
        output = self.fc(last_hidden)  # (batch_size, output_len)
        
        return output
    
    def get_hidden(self, x, pot_ids):
        """获取LSTM的隐藏状态（用于条件预测器）"""
        batch_size, seq_len, _ = x.shape
        
        # 获取槽嵌入并扩展为序列长度
        pot_embed = self.pot_embedding(pot_ids)
        pot_embed = pot_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 拼接特征和槽嵌入
        x = torch.cat([x, pot_embed], dim=-1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后时刻的隐藏状态
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.bn(last_hidden)
        
        return last_hidden  # (batch_size, hidden_dim)


class ConditionalVoltagePredictor(nn.Module):
    """条件电压预测器 - 基于原模型+条件编码器"""
    def __init__(self, base_model, future_action_dim=2, future_len=OUTPUT_LEN, 
                 cond_hidden=32, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.base_model = base_model  # 原LSTM模型，可冻结或微调
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        
        # 条件编码器：LSTM处理未来14天×2个动作，保留时序信息
        self.cond_encoder = nn.LSTM(
            input_size=future_action_dim,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.cond_fc = nn.Linear(16, cond_hidden)
        
        # 合并后的预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + cond_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, future_len)
        )
    
    def forward(self, past_features, future_actions, pot_ids):
        """
        参数:
            past_features: (batch_size, 7, input_dim) 过去7天特征
            future_actions: (batch_size, 14, 2) 未来14天动作序列
            pot_ids: (batch_size,) 槽号索引
        返回:
            output: (batch_size, 14) 未来14天电压预测
        """
        # 获取原模型的隐藏状态
        hidden = self.base_model.get_hidden(past_features, pot_ids)  # (batch_size, hidden_dim)
        
        # 编码未来动作序列（LSTM保留时序信息）
        cond_lstm_out, (cond_h, _) = self.cond_encoder(future_actions)  # (B,14,16), ((1,B,16),...)
        cond = self.cond_fc(cond_h.squeeze(0))  # (batch_size, cond_hidden)
        
        # 拼接历史特征和未来动作特征
        combined = torch.cat([hidden, cond], dim=1)  # (batch_size, hidden_dim + cond_hidden)
        
        # 预测未来14天电压
        output = self.fc(combined)  # (batch_size, 14)
        
        return output
