import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os
from config import *


class VoltageDataset(Dataset):
    def __init__(self, X, y, future_actions=None, pot_ids=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.future_actions = torch.FloatTensor(future_actions) if future_actions is not None else None
        self.pot_ids = torch.LongTensor(pot_ids) if pot_ids is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.pot_ids is not None and self.future_actions is not None:
            return self.X[idx], self.y[idx], self.future_actions[idx], self.pot_ids[idx]
        elif self.pot_ids is not None:
            return self.X[idx], self.y[idx], self.pot_ids[idx]
        return self.X[idx], self.y[idx]


class DataProcessor:
    def __init__(self, data_path=DATA_PATH, input_len=INPUT_LEN, 
                 output_len=OUTPUT_LEN, split_method='time'):
        self.data_path = data_path
        self.input_len = input_len
        self.output_len = output_len
        self.split_method = split_method  # 'time' or 'pot'
        self.scaler = StandardScaler()
        
        # 使用Spearman高相关性特征
        self.all_features = HIGH_CORR_FEATURES
        self.target = TARGET

    def load_data(self):
        df = pd.read_excel(self.data_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['槽号', '日期']).reset_index(drop=True)
        return df

    def preprocess_data(self, df):
        df = df.copy()
        
        # 只保留高相关性特征和必要的列
        # 注意：self.all_features 已包含目标变量，避免重复添加
        all_cols = list(dict.fromkeys(self.all_features + [self.target]))  # 去重并保持顺序
        required_cols = ['日期', '槽号'] + all_cols
        df = df[required_cols]
        
        # 缺失值处理
        numeric_cols = all_cols
        for col in numeric_cols:
            if col in df.columns:
                # 按槽号分组进行前向填充（最多3天）
                df[col] = df.groupby('槽号')[col].transform(lambda x: x.ffill(limit=3))
                # 线性插值
                df[col] = df.groupby('槽号')[col].transform(lambda x: x.interpolate(method='linear'))
                # 剩余缺失值用列均值填充
                df[col] = df[col].fillna(df[col].mean())
        
        return df

    def create_features(self, df):
        df = df.copy()
        feature_cols = self.all_features.copy()
        
        # 为每个高相关性特征创建统计特征和差分特征（不包括目标变量）
        for col in self.all_features:
            if col == self.target:
                continue
                
            # 3天滑动窗口统计特征
            df[f'{col}_mean_3d'] = df.groupby('槽号')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean())
            df[f'{col}_std_3d'] = df.groupby('槽号')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
            
            # 7天滑动窗口统计特征
            df[f'{col}_mean_7d'] = df.groupby('槽号')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean())
            df[f'{col}_std_7d'] = df.groupby('槽号')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
            
            # 1阶差分
            df[f'{col}_diff_1'] = df.groupby('槽号')[col].transform(
                lambda x: x.diff().fillna(0))
            
            # 7阶差分
            df[f'{col}_diff_7'] = df.groupby('槽号')[col].transform(
                lambda x: x.diff(7).fillna(0))
            
            # 添加新特征列名
            feature_cols.extend([
                f'{col}_mean_3d', f'{col}_std_3d',
                f'{col}_mean_7d', f'{col}_std_7d',
                f'{col}_diff_1', f'{col}_diff_7'
            ])
        
        # 为目标变量创建统计特征（用于输入序列）
        # 注意：目标变量已在 HIGH_CORR_FEATURES 中，但循环中已跳过
        # 这里单独为目标变量创建统计特征
        df[f'{self.target}_mean_3d'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{self.target}_std_3d'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
        df[f'{self.target}_mean_7d'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df[f'{self.target}_std_7d'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
        df[f'{self.target}_diff_1'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.diff().fillna(0))
        df[f'{self.target}_diff_7'] = df.groupby('槽号')[self.target].transform(
            lambda x: x.diff(7).fillna(0))
        
        feature_cols.extend([
            f'{self.target}_mean_3d', f'{self.target}_std_3d',
            f'{self.target}_mean_7d', f'{self.target}_std_7d',
            f'{self.target}_diff_1', f'{self.target}_diff_7'
        ])
        
        # 衍生特征
        if '工作平均' in df.columns and '电压设定' in df.columns:
            df['电压偏差'] = df['工作平均'] - df['电压设定']
            feature_cols.append('电压偏差')
        
        if '铝水平' in df.columns and '电解质水平' in df.columns:
            df['铝电解比例'] = df['铝水平'] / (df['电解质水平'] + 1e-8)
            feature_cols.append('铝电解比例')
        
        # 槽龄相关特征（非线性处理）
        if '槽龄' in df.columns:
            df['槽龄_log'] = np.log1p(df['槽龄'])
            df['槽龄_squared'] = df['槽龄'] ** 2
            feature_cols.extend(['槽龄_log', '槽龄_squared'])
        
        # 填充任何剩余的NaN值
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        return df, feature_cols

    def create_sequences(self, df, feature_cols, use_future_actions=False):
        X, y, future_actions, pot_ids = [], [], [], []
        
        # 动作特征列名
        action_features = ['ALF加料量(实际)', '实际出铝量']
        
        # 为每个槽号创建序列
        for pot_id in df['槽号'].unique():
            pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
            
            required_days = self.input_len + self.output_len
            if len(pot_data) < required_days:
                print(f"警告: 槽号 {pot_id} 只有 {len(pot_data)} 天数据，需要至少 {required_days} 天，已跳过")
                continue
            
            pot_features = pot_data[feature_cols].values
            pot_target = pot_data[self.target].values
            
            # 提取动作数据
            pot_alf = pot_data['ALF加料量(实际)'].values
            pot_out = pot_data['实际出铝量'].values
            
            # 创建滑动窗口序列
            for i in range(len(pot_data) - self.input_len - self.output_len + 1):
                X.append(pot_features[i:i + self.input_len])
                y.append(pot_target[i + self.input_len:i + self.input_len + self.output_len])
                pot_ids.append(pot_id)
                
                # 提取未来14天的动作序列（如果启用）
                if use_future_actions:
                    # 未来14天的ALF加料量和实际出铝量
                    future_alf = pot_alf[i + self.input_len:i + self.input_len + self.output_len]
                    future_out = pot_out[i + self.input_len:i + self.input_len + self.output_len]
                    
                    # 组合成 (14, 2) 的形状
                    future_action_seq = np.stack([future_alf, future_out], axis=1)
                    future_actions.append(future_action_seq)
        
        if len(X) == 0:
            print(f"警告: 没有足够的数据创建序列！需要至少 {required_days} 天/槽")
        
        if use_future_actions:
            return np.array(X), np.array(y), np.array(future_actions), np.array(pot_ids)
        else:
            return np.array(X), np.array(y), np.array(pot_ids)

    def split_data(self, X, y, pot_ids, df, future_actions=None):
        if self.split_method == 'time':
            # 按时间顺序划分（70%训练，17%验证，13%测试）
            n_samples = len(X)
            train_end = int(n_samples * 0.7)
            val_end = int(n_samples * 0.87)
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
            
            pot_ids_train = pot_ids[:train_end]
            pot_ids_val = pot_ids[train_end:val_end]
            pot_ids_test = pot_ids[val_end:]
            
            if future_actions is not None:
                future_actions_train = future_actions[:train_end]
                future_actions_val = future_actions[train_end:val_end]
                future_actions_test = future_actions[val_end:]
            else:
                future_actions_train = future_actions_val = future_actions_test = None
            
        else:  # 'pot'
            # 按槽号划分
            train_mask = np.isin(pot_ids, TRAIN_POTS)
            val_mask = np.isin(pot_ids, VAL_POTS)
            test_mask = np.isin(pot_ids, TEST_POTS)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            pot_ids_train = pot_ids[train_mask]
            pot_ids_val = pot_ids[val_mask]
            pot_ids_test = pot_ids[test_mask]
            
            if future_actions is not None:
                future_actions_train = future_actions[train_mask]
                future_actions_val = future_actions[val_mask]
                future_actions_test = future_actions[test_mask]
            else:
                future_actions_train = future_actions_val = future_actions_test = None
        
        return (X_train, y_train, pot_ids_train, future_actions_train), \
               (X_val, y_val, pot_ids_val, future_actions_val), \
               (X_test, y_test, pot_ids_test, future_actions_test)

    def process(self, use_future_actions=False):
        print("=" * 60)
        print("开始数据处理")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        df = self.load_data()
        print(f"   原始数据形状: {df.shape}")
        print(f"   槽号数量: {df['槽号'].nunique()}")
        print(f"   日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
        
        # 2. 数据预处理
        print("\n2. 数据预处理...")
        df = self.preprocess_data(df)
        print("   缺失值处理完成")
        
        # 3. 特征工程
        print("\n3. 特征工程...")
        df, feature_cols = self.create_features(df)
        print(f"   特征数量: {len(feature_cols)}")
        print(f"   特征列表: {feature_cols[:5]}... (共{len(feature_cols)}个)")
        
        # 4. 数据集划分（先划分，再标准化，避免数据泄露）
        print("\n4. 数据集划分...")
        
        df_train = []
        df_val = []
        df_test = []
        
        required_days = self.input_len + self.output_len
        
        if self.split_method == 'time':
            # 按时间划分：每个槽号的数据按时间顺序划分为训练/验证/测试
            print(f"   划分方式: 时间划分")
            
            for pot_id in df['槽号'].unique():
                pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
                total_days = len(pot_data)
                
                # 严格要求：必须有足够数据支持三个独立的数据集
                if total_days < 3 * required_days:
                    print(f"   跳过槽号 {pot_id}: 只有 {total_days} 天数据，需要至少 {3 * required_days} 天")
                    continue
                
                # 数据充足时，按比例划分
                train_end = int(total_days * 0.6)
                val_end = int(total_days * 0.8)
                
                # 确保验证集和测试集至少有 required_days 天
                val_end = max(val_end, train_end + required_days)
                val_end = min(val_end, total_days - required_days)
                
                df_train.append(pot_data.iloc[:train_end])
                df_val.append(pot_data.iloc[train_end:val_end])
                df_test.append(pot_data.iloc[val_end:])
            
            if len(df_train) == 0:
                raise ValueError(f"错误: 没有槽号有足够的数据（至少需要 {3 * required_days} 天）！")
        
        else:
            # 按槽号划分：使用预定义的槽号分组
            print(f"   划分方式: 槽号划分")
            print(f"   训练槽号数量: {len(TRAIN_POTS)}")
            print(f"   验证槽号数量: {len(VAL_POTS)}")
            print(f"   测试槽号数量: {len(TEST_POTS)}")
            
            for pot_id in df['槽号'].unique():
                pot_data = df[df['槽号'] == pot_id].sort_values('日期').reset_index(drop=True)
                total_days = len(pot_data)
                
                # 每个槽号至少需要 required_days 天数据
                if total_days < required_days:
                    print(f"   跳过槽号 {pot_id}: 只有 {total_days} 天数据，需要至少 {required_days} 天")
                    continue
                
                # 根据槽号分组分配
                if pot_id in TRAIN_POTS:
                    df_train.append(pot_data)
                elif pot_id in VAL_POTS:
                    df_val.append(pot_data)
                elif pot_id in TEST_POTS:
                    df_test.append(pot_data)
            
            if len(df_train) == 0:
                raise ValueError("错误: 训练槽号中没有足够数据的槽号！")
            if len(df_val) == 0:
                raise ValueError("错误: 验证槽号中没有足够数据的槽号！")
            if len(df_test) == 0:
                raise ValueError("错误: 测试槽号中没有足够数据的槽号！")
        
        df_train = pd.concat(df_train, ignore_index=True)
        df_val = pd.concat(df_val, ignore_index=True)
        df_test = pd.concat(df_test, ignore_index=True)
        
        print(f"   训练集: {len(df_train)} 条记录")
        print(f"   验证集: {len(df_val)} 条记录")
        print(f"   测试集: {len(df_test)} 条记录")
        print(f"   ✅ 验证集与测试集完全独立，无数据共享")
        
        # 5. 数据标准化（在训练集上拟合，然后转换所有数据集）
        print("\n5. 数据标准化...")
        X_train_features = df_train[feature_cols].values
        self.scaler.fit(X_train_features)
        
        df_train[feature_cols] = self.scaler.transform(df_train[feature_cols].values)
        df_val[feature_cols] = self.scaler.transform(df_val[feature_cols].values)
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols].values)
        
        # 6. 创建序列
        print("\n6. 创建序列...")
        if use_future_actions:
            X_train, y_train, future_actions_train, pot_ids_train = self.create_sequences(df_train, feature_cols, use_future_actions=True)
            X_val, y_val, future_actions_val, pot_ids_val = self.create_sequences(df_val, feature_cols, use_future_actions=True)
            X_test, y_test, future_actions_test, pot_ids_test = self.create_sequences(df_test, feature_cols, use_future_actions=True)
        else:
            X_train, y_train, pot_ids_train = self.create_sequences(df_train, feature_cols)
            X_val, y_val, pot_ids_val = self.create_sequences(df_val, feature_cols)
            X_test, y_test, pot_ids_test = self.create_sequences(df_test, feature_cols)
            future_actions_train = future_actions_val = future_actions_test = None
        
        print(f"   训练序列: {len(X_train)} 样本")
        print(f"   验证序列: {len(X_val)} 样本")
        print(f"   测试序列: {len(X_test)} 样本")
        print(f"   输入形状: {X_train.shape}")
        print(f"   输出形状: {y_train.shape}")
        if use_future_actions:
            print(f"   未来动作形状: {future_actions_train.shape}")
        
        # 7. 创建槽号映射
        all_pots = np.unique(np.concatenate([pot_ids_train, pot_ids_val, pot_ids_test]))
        pot_id_mapping = {pot: idx for idx, pot in enumerate(sorted(all_pots))}
        
        pot_ids_train_mapped = np.array([pot_id_mapping[pot] for pot in pot_ids_train])
        pot_ids_val_mapped = np.array([pot_id_mapping[pot] for pot in pot_ids_val])
        pot_ids_test_mapped = np.array([pot_id_mapping[pot] for pot in pot_ids_test])
        
        print(f"   槽号映射: {len(pot_id_mapping)} 个槽号")
        
        # 8. 创建DataLoader
        print("\n7. 创建DataLoader...")
        if use_future_actions:
            train_dataset = VoltageDataset(X_train, y_train, future_actions_train, pot_ids_train_mapped)
            val_dataset = VoltageDataset(X_val, y_val, future_actions_val, pot_ids_val_mapped)
            test_dataset = VoltageDataset(X_test, y_test, future_actions_test, pot_ids_test_mapped)
        else:
            train_dataset = VoltageDataset(X_train, y_train, pot_ids=pot_ids_train_mapped)
            val_dataset = VoltageDataset(X_val, y_val, pot_ids=pot_ids_val_mapped)
            test_dataset = VoltageDataset(X_test, y_test, pot_ids=pot_ids_test_mapped)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print("   DataLoader创建完成")
        print("=" * 60)
        
        # 保存scaler
        with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return train_loader, val_loader, test_loader, len(pot_id_mapping), feature_cols
