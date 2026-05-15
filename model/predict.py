import torch
import numpy as np
import pandas as pd
import pickle
import os
import argparse

from model import LSTMModelWithPotEmbedding
from config import *


def load_model_and_scaler(model_path, scaler_path, input_dim, num_pots):
    """加载训练好的模型和标准化器"""
    model = LSTMModelWithPotEmbedding(
        input_dim=input_dim,
        num_pots=num_pots,
        pot_embed_dim=POT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_len=OUTPUT_LEN,
        dropout=DROPOUT
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_voltage(model, scaler, recent_data, pot_id, pot_id_mapping):
    """
    预测未来工作电压
    
    参数:
        model: 训练好的模型
        scaler: 标准化器
        recent_data: 最近7天的特征数据 (7, input_dim)
        pot_id: 槽号（原始槽号）
        pot_id_mapping: 槽号到整数ID的映射字典
    
    返回:
        prediction: 未来工作电压预测值 (array of shape (output_len,))
    """
    # 标准化
    data_scaled = scaler.transform(recent_data)
    
    # 转换为张量
    X = torch.FloatTensor(data_scaled).unsqueeze(0)
    pot_ids = torch.LongTensor([pot_id_mapping[pot_id]])
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(X, pot_ids)
    
    return output.squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description='工作电压预测')
    parser.add_argument('--input', type=str, help='输入数据文件路径（CSV格式）')
    parser.add_argument('--output', type=str, default='predictions.csv', help='输出预测结果文件路径')
    parser.add_argument('--model', type=str, default=os.path.join(OUTPUT_DIR, 'best_model.pth'), 
                        help='模型权重文件路径')
    parser.add_argument('--scaler', type=str, default=os.path.join(OUTPUT_DIR, 'scaler.pkl'), 
                        help='标准化器文件路径')
    
    args = parser.parse_args()
    
    # 槽号映射（与训练时保持一致）
    all_pots = TRAIN_POTS + VAL_POTS + TEST_POTS
    pot_id_mapping = {pot: idx for idx, pot in enumerate(sorted(all_pots))}
    num_pots = len(all_pots)
    
    if args.input:
        # 从文件读取数据
        df = pd.read_csv(args.input)
        
        # 提取特征（排除日期和槽号）
        features = df.drop(['槽号', '日期'], axis=1, errors='ignore').values
        
        if features.shape[0] >= 7:
            recent_data = features[-7:]  # 取最近7天
            pot_id = df['槽号'].iloc[-1]
            
            # 获取特征维度
            input_dim = recent_data.shape[1]
            
            # 加载模型和scaler
            model, scaler = load_model_and_scaler(args.model, args.scaler, input_dim, num_pots)
            
            # 预测
            predictions = predict_voltage(model, scaler, recent_data, pot_id, pot_id_mapping)
            
            # 保存结果
            result_df = pd.DataFrame({
                '预测天数': range(1, len(predictions) + 1),
                '工作电压预测值': predictions
            })
            result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存至: {args.output}")
            print(f"\n预测的未来{len(predictions)}天工作电压:")
            for i, pred in enumerate(predictions, 1):
                print(f"  Day {i}: {pred:.4f}")
        else:
            print("输入数据不足7天，无法进行预测")
    else:
        # 演示模式：生成模拟数据进行预测
        print("演示模式：使用模拟数据进行预测")
        
        # 加载scaler以获取特征维度
        with open(args.scaler, 'rb') as f:
            scaler = pickle.load(f)
        input_dim = scaler.n_features_in_
        
        # 加载模型
        model, _ = load_model_and_scaler(args.model, args.scaler, input_dim, num_pots)
        
        # 生成模拟数据
        recent_data = np.random.randn(7, input_dim)  # 模拟7天的特征数据
        pot_id = 1101
        
        predictions = predict_voltage(model, scaler, recent_data, pot_id, pot_id_mapping)
        
        print(f"\n未来{len(predictions)}天工作电压预测结果:")
        for i, pred in enumerate(predictions, 1):
            print(f"  Day {i}: {pred:.4f}")


if __name__ == '__main__':
    main()
