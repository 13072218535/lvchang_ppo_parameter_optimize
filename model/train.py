import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

from data_processor import DataProcessor
from model import LSTMModel, LSTMModelWithPotEmbedding, ConditionalVoltagePredictor
from config import *


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_by_day(y_true, y_pred):
    """按天评估多步预测结果"""
    metrics_by_day = []
    for day in range(y_true.shape[1]):
        day_true = y_true[:, day]
        day_pred = y_pred[:, day]
        metrics = calculate_metrics(day_true, day_pred)
        metrics_by_day.append(metrics)
    return metrics_by_day


def train_epoch(model, train_loader, criterion, optimizer, device, use_pot_embedding, use_conditional=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        if use_conditional:
            # 条件预测模型：输入包含未来动作
            X_batch, y_batch, future_actions_batch, pot_ids_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            future_actions_batch = future_actions_batch.to(device)
            pot_ids_batch = pot_ids_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, future_actions_batch, pot_ids_batch)
        elif use_pot_embedding:
            X_batch, y_batch, pot_ids_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pot_ids_batch = pot_ids_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, pot_ids_batch)
        else:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, use_pot_embedding, use_conditional=False):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if use_conditional:
                # 条件预测模型：输入包含未来动作
                X_batch, y_batch, future_actions_batch, pot_ids_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                future_actions_batch = future_actions_batch.to(device)
                pot_ids_batch = pot_ids_batch.to(device)
                outputs = model(X_batch, future_actions_batch, pot_ids_batch)
            elif use_pot_embedding:
                X_batch, y_batch, pot_ids_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pot_ids_batch = pot_ids_batch.to(device)
                outputs = model(X_batch, pot_ids_batch)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def test_evaluate(model, test_loader, device, use_pot_embedding, use_conditional=False):
    """测试集评估"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if use_conditional:
                # 条件预测模型：输入包含未来动作
                X_batch, y_batch, future_actions_batch, pot_ids_batch = batch
                X_batch = X_batch.to(device)
                future_actions_batch = future_actions_batch.to(device)
                pot_ids_batch = pot_ids_batch.to(device)
                outputs = model(X_batch, future_actions_batch, pot_ids_batch)
            elif use_pot_embedding:
                X_batch, y_batch, pot_ids_batch = batch
                X_batch = X_batch.to(device)
                pot_ids_batch = pot_ids_batch.to(device)
                outputs = model(X_batch, pot_ids_batch)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                outputs = model(X_batch)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    return y_true, y_pred


def plot_losses(train_losses, val_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_losses) + 1)
    plt.semilogy(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.semilogy(epochs, val_losses, label='Val Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE, log scale)', fontsize=12)
    plt.title('Training and Validation Loss (Log Scale)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存至: {save_path}")


def plot_metrics_by_day(metrics_by_day, save_path):
    """绘制分日评估指标"""
    days = list(range(1, len(metrics_by_day) + 1))
    mae = [m['MAE'] for m in metrics_by_day]
    rmse = [m['RMSE'] for m in metrics_by_day]
    mape = [m['MAPE'] for m in metrics_by_day]
    r2 = [m['R2'] for m in metrics_by_day]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.bar(days, mae, color='steelblue', alpha=0.8)
    plt.xlabel('Prediction Day', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE by Prediction Day', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 4, 2)
    plt.bar(days, rmse, color='coral', alpha=0.8)
    plt.xlabel('Prediction Day', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE by Prediction Day', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 4, 3)
    plt.bar(days, mape, color='seagreen', alpha=0.8)
    plt.xlabel('Prediction Day', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.title('MAPE by Prediction Day', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 4, 4)
    plt.bar(days, r2, color='purple', alpha=0.8)
    plt.xlabel('Prediction Day', fontsize=12)
    plt.ylabel('R2', fontsize=12)
    plt.title('R2 by Prediction Day', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分日评估指标已保存至: {save_path}")


def plot_predictions(y_true, y_pred, num_samples=3, save_path=None):
    """绘制多步预测对比图（样本示例）"""
    plt.figure(figsize=(18, 10))

    indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 1, i + 1)
        days = range(1, y_true.shape[1] + 1)
        plt.plot(days, y_true[idx], 'b-o', label='True', linewidth=2, markersize=5)
        plt.plot(days, y_pred[idx], 'r--s', label='Predicted', linewidth=2, markersize=5)
        plt.fill_between(days, y_true[idx], y_pred[idx],
                        where=y_pred[idx] >= y_true[idx], facecolor='pink', alpha=0.3,
                        interpolate=True)
        plt.fill_between(days, y_true[idx], y_pred[idx],
                        where=y_pred[idx] < y_true[idx], facecolor='lightblue', alpha=0.3,
                        interpolate=True)
        plt.xlabel('Day')
        plt.ylabel('Working Voltage')
        plt.title(f'Sample {i + 1}: {len(days)}-Day Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(days) + 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"多步预测对比图已保存至: {save_path}")


def plot_all_test_predictions_scatter(y_true, y_pred, save_path=None):
    """绘制所有测试集数据的真实值与预测值散点对比图"""
    # 将多步预测展平为一维数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    plt.figure(figsize=(12, 10))
    
    # 计算最优拟合线
    z = np.polyfit(y_true_flat, y_pred_flat, 1)
    p = np.poly1d(z)
    
    # 计算统计指标
    mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))
    
    # 散点图
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, color='steelblue', edgecolor='white', s=50)
    
    # 对角线（理想预测）
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 拟合线
    plt.plot(y_true_flat, p(y_true_flat), 'g--', linewidth=2, 
             label=f'Fitted Line: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    plt.xlabel('True Working Voltage', fontsize=14)
    plt.ylabel('Predicted Working Voltage', fontsize=14)
    plt.title(f'Test Set Predictions vs True Values (n={len(y_true_flat)})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nSamples: {len(y_true_flat)}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"测试集全量散点对比图已保存至: {save_path}")


def plot_all_test_predictions_timeseries(y_true, y_pred, save_path=None):
    """绘制所有测试集数据的时序对比图（参考电流预测样式）"""
    # 将所有样本的预测结果展开为完整的时间序列
    y_true_series = y_true.flatten()
    y_pred_series = y_pred.flatten()
    
    # 计算统计指标
    mae = np.mean(np.abs(y_pred_series - y_true_series))
    rmse = np.sqrt(np.mean((y_pred_series - y_true_series) ** 2))
    
    plt.figure(figsize=(20, 6))
    
    # 时间点（使用样本索引作为横坐标）
    num_samples = y_true.shape[0]
    num_days = y_true.shape[1]
    time_points = np.arange(len(y_true_series))
    
    # 绘制真实值（蓝色实线）
    plt.plot(time_points, y_true_series, 'b-', label='True', linewidth=1)
    
    # 绘制预测值（红色虚线）
    plt.plot(time_points, y_pred_series, 'r--', label='Predicted', linewidth=1)
    
    # 填充误差区域
    plt.fill_between(time_points, y_true_series, y_pred_series,
                    where=y_pred_series >= y_true_series, facecolor='pink', alpha=0.2,
                    interpolate=True)
    plt.fill_between(time_points, y_true_series, y_pred_series,
                    where=y_pred_series < y_true_series, facecolor='lightblue', alpha=0.2,
                    interpolate=True)
    
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Working Voltage', fontsize=12)
    plt.title(f'Test Set: All Predictions vs True Values (Samples={num_samples}, Days={num_days})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.02, 0.98, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"测试集全量时序对比图已保存至: {save_path}")


def moving_average(data, window_size=3):
    """计算移动平均值，使曲线更平滑"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def plot_test_predictions_combined(y_true, y_pred, save_path=None):
    """绘制测试集预测对比图（组合版：时序+散点）- 美化版"""
    plt.figure(figsize=(20, 12))
    
    # 子图1：时序对比（展示前200个时间点）
    plt.subplot(2, 1, 1)
    y_true_series = y_true.flatten()
    y_pred_series = y_pred.flatten()
    
    # 只展示前200个点便于观察
    show_points = min(200, len(y_true_series))
    time_points = range(1, show_points + 1)
    
    # 计算移动平均使曲线更平滑
    smooth_true = moving_average(y_true_series[:show_points], window_size=3)
    smooth_pred = moving_average(y_pred_series[:show_points], window_size=3)
    
    # 使用更美观的颜色和线条样式
    plt.plot(time_points, smooth_true, color='#1f77b4', label='True', linewidth=2.5, linestyle='-', alpha=0.9)
    plt.plot(time_points, smooth_pred, color='#ff7f0e', label='Predicted', linewidth=2.5, linestyle='--', alpha=0.9)
    
    # 填充误差区域（使用渐变效果）
    plt.fill_between(time_points, smooth_true, smooth_pred,
                    where=smooth_pred >= smooth_true, 
                    facecolor='#ff9896', alpha=0.25, interpolate=True)
    plt.fill_between(time_points, smooth_true, smooth_pred,
                    where=smooth_pred < smooth_true, 
                    facecolor='#7f7fff', alpha=0.25, interpolate=True)
    
    plt.xlabel('Time Point (First 200)', fontsize=14, fontweight='bold')
    plt.ylabel('Working Voltage', fontsize=14, fontweight='bold')
    plt.title('Test Set Predictions vs True Values (Time Series View)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tick_params(axis='both', labelsize=12)
    
    # 子图2：散点对比
    plt.subplot(2, 1, 2)
    plt.scatter(y_true_series, y_pred_series, alpha=0.4, color='#2ca02c', edgecolor='white', s=40, linewidth=0.5)
    
    min_val = min(y_true_series.min(), y_pred_series.min())
    max_val = max(y_true_series.max(), y_pred_series.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#ff7f0e', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    
    plt.xlabel('True Working Voltage', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Working Voltage', fontsize=14, fontweight='bold')
    plt.title('Test Set Predictions vs True Values (Scatter View)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout(pad=3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"测试集预测对比图（组合版）已保存至: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_augmented', action='store_true', help='Include adversarial augmented data')
    args = parser.parse_args()

    # 设置随机种子（保证实验可复现）
    set_seed(SEED)
    print(f"已设置随机种子: {SEED}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 参数配置
    use_pot_embedding = True
    use_conditional = True  # 是否使用条件预测模型
    split_method = 'pot'

    # 增强数据路径
    aug_path = os.path.join(OUTPUT_DIR, 'adversarial_augmented.pkl') if args.use_augmented else None
    if args.use_augmented:
        print(f"增强数据: {aug_path} (exists={os.path.exists(aug_path)})")

    print("=" * 60)
    print("工作电压预测 - 模型训练")
    print("=" * 60)

    # 数据处理
    processor = DataProcessor(data_path=DATA_PATH, input_len=INPUT_LEN,
                              output_len=OUTPUT_LEN, split_method=split_method)
    train_loader, val_loader, test_loader, num_pots, feature_cols = \
        processor.process(use_future_actions=use_conditional, augmented_data_path=aug_path)

    # 获取特征数量
    num_features = train_loader.dataset.X.shape[-1]

    print("\n模型配置:")
    print(f"- 使用槽嵌入: {use_pot_embedding}")
    print(f"- 使用条件预测: {use_conditional}")
    print(f"- 槽数量: {num_pots}")
    print(f"- 特征数量: {num_features}")
    print(f"- 输入序列长度: {INPUT_LEN}")
    print(f"- 输出序列长度: {OUTPUT_LEN}")

    # 创建模型
    if use_conditional:
        # 条件预测模型：先创建基础LSTM模型，然后包装为条件预测器
        base_model = LSTMModelWithPotEmbedding(
            input_dim=num_features,
            num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_len=OUTPUT_LEN,
            dropout=DROPOUT
        ).to(device)
        
        # 加载预训练的基础模型权重
        pretrained_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(pretrained_path):
            try:
                base_model.load_state_dict(torch.load(pretrained_path))
                print(f"\n加载预训练基础模型权重: {pretrained_path}")
                for param in base_model.parameters():
                    param.requires_grad = True
                print("基础模型权重已加载（差分学习率模式，基础LSTM可微调）")
            except RuntimeError as e:
                print(f"\n⚠ 预训练模型架构不兼容，将从头训练基础LSTM")
                print(f"   原因: {str(e)[:120]}...")
                print(f"   请先运行: python train.py (设置 use_conditional=False)")
                print(f"   训练基础LSTM后再运行: python train.py (设置 use_conditional=True)")
                # 删除不兼容的旧模型文件
                os.remove(pretrained_path)
                print(f"   已删除旧模型: {pretrained_path}")
        else:
            print(f"\n⚠ 预训练基础模型不存在于 {pretrained_path}")
            print("   请先设置 use_conditional=False 并运行 train.py 训练基础LSTM")
            print("   基础LSTM将以随机权重初始化，预测精度将严重受限")
        
        # 创建条件预测器
        model = ConditionalVoltagePredictor(
            base_model=base_model,
            future_action_dim=2,
            future_len=OUTPUT_LEN,
            cond_hidden=32,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        ).to(device)
    elif use_pot_embedding:
        model = LSTMModelWithPotEmbedding(
            input_dim=num_features,
            num_pots=num_pots,
            pot_embed_dim=POT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_len=OUTPUT_LEN,
            dropout=DROPOUT
        ).to(device)
    else:
        model = LSTMModel(
            input_dim=num_features,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_len=OUTPUT_LEN,
            dropout=DROPOUT
        ).to(device)

    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, 
                                  patience=REDUCE_LR_PATIENCE)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 根据模型类型设置不同的保存路径
    if use_conditional:
        model_save_path = os.path.join(OUTPUT_DIR, 'best_conditional_model.pth')
        final_model_path = os.path.join(OUTPUT_DIR, 'final_conditional_model.pth')
    else:
        model_save_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"模型权重将保存至: {model_save_path}")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_pot_embedding, use_conditional)
        val_loss = validate_epoch(model, val_loader, criterion, device, use_pot_embedding, use_conditional)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch + 1:03d}] 验证损失下降，已保存模型权重")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} "
                  f"| Best: {best_val_loss:.6f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发于 Epoch {epoch + 1}")
            break

    print("\n训练完成!")
    
    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    print(f"已保存最终模型权重至: {final_model_path}")

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    print(f"已加载最佳模型权重: {model_save_path}")

    # 绘制损失曲线
    plot_losses(train_losses, val_losses, os.path.join(OUTPUT_DIR, 'loss_curves.png'))

    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    y_true, y_pred = test_evaluate(model, test_loader, device, use_pot_embedding, use_conditional)

    # 计算评估指标
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    overall_metrics = calculate_metrics(y_true_flat, y_pred_flat)
    print("\n测试集整体评估指标:")
    print("-" * 40)
    for metric_name, value in overall_metrics.items():
        print(f"{metric_name}: {value:.6f}")

    # 按天评估（多步预测专用）
    if OUTPUT_LEN > 1:
        metrics_by_day = evaluate_by_day(y_true, y_pred)
        print("\n各预测日评估指标:")
        print("-" * 40)
        for day, metrics in enumerate(metrics_by_day, 1):
            print(f"Day {day}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.4f}%, R2={metrics['R2']:.4f}")
        
        # 绘制分日评估图
        plot_metrics_by_day(metrics_by_day, os.path.join(OUTPUT_DIR, 'metrics_by_day.png'))

    # 绘制预测对比图（多步预测样本）
    plot_predictions(y_true, y_pred, num_samples=3, 
                     save_path=os.path.join(OUTPUT_DIR, 'predictions_comparison.png'))
    
    # 绘制所有测试集数据的散点对比图
    plot_all_test_predictions_scatter(y_true, y_pred, 
                                      save_path=os.path.join(OUTPUT_DIR, 'all_test_predictions_scatter.png'))
    
    # 绘制所有测试集数据的时序对比图（参考电流预测样式）
    plot_all_test_predictions_timeseries(y_true, y_pred, 
                                         save_path=os.path.join(OUTPUT_DIR, 'all_test_predictions_timeseries.png'))
    
    # 绘制组合版对比图（时序+散点）
    plot_test_predictions_combined(y_true, y_pred, 
                                   save_path=os.path.join(OUTPUT_DIR, 'test_predictions_combined.png'))

    # 保存测试结果
    results_file = os.path.join(OUTPUT_DIR, 'test_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("工作电压预测测试结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"参数配置:\n")
        f.write(f"  - 使用槽嵌入: {use_pot_embedding}\n")
        f.write(f"  - 分割方式: {split_method}\n")
        f.write(f"  - 特征数量: {len(feature_cols)}\n")
        f.write(f"  - 槽数量: {num_pots}\n")
        f.write(f"  - 输入序列长度: {INPUT_LEN}\n")
        f.write(f"  - 输出序列长度: {OUTPUT_LEN}\n\n")

        f.write("整体评估指标:\n")
        f.write("-" * 40 + "\n")
        for metric_name, value in overall_metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")

        if OUTPUT_LEN > 1:
            f.write("\n各预测日评估指标:\n")
            f.write("-" * 40 + "\n")
            for day, metrics in enumerate(metrics_by_day, 1):
                f.write(f"Day {day}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                       f"MAPE={metrics['MAPE']:.4f}%, R2={metrics['R2']:.4f}\n")

        f.write("\n特征列表:\n")
        f.write("-" * 40 + "\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"{i:>3}. {feat}\n")

    print(f"\n测试结果已保存至: {results_file}")

    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)
    print(f"\n输出文件目录: {OUTPUT_DIR}")
    print("  - best_model.pth: 最佳模型权重")
    print("  - scaler.pkl: 标准化器对象")
    print("  - loss_curves.png: 训练验证损失曲线")
    print("  - predictions_comparison.png: 预测对比图（样本）")
    print("  - all_test_predictions_scatter.png: 测试集全量散点对比图")
    print("  - all_test_predictions_timeseries.png: 测试集全量时序对比图")
    print("  - test_predictions_combined.png: 测试集组合对比图")
    if OUTPUT_LEN > 1:
        print("  - metrics_by_day.png: 分日评估指标")
    print("  - test_results.txt: 测试结果文本")


if __name__ == '__main__':
    main()
