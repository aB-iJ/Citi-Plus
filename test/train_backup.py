import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import os

from config import config
from data_loader import get_processed_data
from model import AttentionBiGRU
from utils import get_device, set_seed

# Gaussian NLL Loss for Regression with Uncertainty (高斯负对数似然损失函数)
def gaussian_nll_loss(pred_mu, log_var, target):
    # 限制 log_var 范围以保证数值稳定性
    log_var = torch.clamp(log_var, min=-10, max=10)
    precision = torch.exp(-log_var)
    loss = 0.5 * precision * (pred_mu - target)**2 + 0.5 * log_var
    return loss.mean()

def direction_loss(pred_diff, actual_diff):
    # 强化版方向损失
    # 1. 基础方向惩罚: relu(-prod)
    basic_loss = torch.relu(-1.0 * pred_diff * actual_diff)
    
    # 2. 符号惩罚 (Sign Penalty)
    # 使用 soft sign (tanh) 来近似符号函数
    # 如果两个 tanh 符号相反，产生的 Loss 较大
    sign_loss = torch.relu( -1.0 * torch.tanh(pred_diff * 10) * torch.tanh(actual_diff * 10) )
    
    return basic_loss.mean() + sign_loss.mean()

def create_sequences(data, seq_length, target_cols):
    xs = []
    ys = []
    
    n_features = data.shape[1] - len(target_cols)
    
    for i in range(len(data) - seq_length):
        x_seq = data[i : i+seq_length, :n_features]
        y_label = data[i+seq_length-1, n_features:]
        xs.append(x_seq)
        ys.append(y_label)
        
    return np.array(xs), np.array(ys)

def train():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. 加载数据
    df = get_processed_data()
    
    # [关键修改] 选择纯平稳特征 (Stationary Features)
    # 排除所有绝对价格列 (Oil_Close, SMA_5, etc.)，只保留 Ratio/Diff/Oscillator
    stationary_cols = [
        'Dist_SMA_5', 'Dist_SMA_20', 
        'RSI', 'MACD', 
        'ROC_1', 'ROC_3', 'ROC_5', 
        # [新增] 二阶导数特征 - 加速度
        'ROC_Accel_1', 'ROC_Accel_3', 'RSI_Momentum',
        'Bollinger_PctB', 'Bollinger_Width',
        'News_Impact',
        'Oil_Volume'
    ]
    # 添加因子收益率
    for t in config.TICKERS_FACTORS:
        ret_col = f"{t}_Ret"
        if ret_col in df.columns:
            stationary_cols.append(ret_col)
            
    # 二次确认这些列存在
    feature_cols = [c for c in stationary_cols if c in df.columns]
    
    # 多任务目标: 收益率 + 波动率 + 涨跌方向(分类)
    target_cols = ["Target_Return", "Target_Volatility", "Target_Direction"]
    
    print(f"Features (Stationary, Count={len(feature_cols)}): {feature_cols}")
    
    # 初始化 Standard Scaler
    scaler = StandardScaler()
    scaler_target = StandardScaler()

    # [关键修复] 数据泄漏问题
    # 必须仅使用训练集来拟合 Scaler，否则测试集的分布信息会泄漏给模型
    # 我们按时间切分为 80% 训练, 20% 验证
    border_idx = int(len(df) * 0.8)
    
    # 分割 DataFrame
    train_df = df.iloc[:border_idx]
    
    # Fit Scaler only on Train
    scaler.fit(train_df[feature_cols])
    scaler_target.fit(train_df[target_cols])
    
    # Transform whole dataset
    data_scaled = scaler.transform(df[feature_cols])
    target_scaled = scaler_target.transform(df[target_cols])
    
    # 保存 Scalers 以便后续推理使用
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_features.pkl")
    joblib.dump(scaler_target, "models/scaler_targets.pkl")
    joblib.dump(feature_cols, "models/feature_names.pkl")
    
    # 合并数据以创建时间序列
    combined_data = np.hstack([data_scaled, target_scaled])
    
    X, y = create_sequences(combined_data, config.SEQ_LENGTH, target_cols)
    print(f"Total Sequences: {X.shape[0]}")
    
    # 转换为 Tensor
    X_to = torch.FloatTensor(X).to(device)
    y_to = torch.FloatTensor(y).to(device)
    
    # 重新计算切分点 (因为 create_sequences 减少了长度)
    # 保持之前基于原始数据 80% 的时间点
    # create_sequences 移除了前 SEQ_LENGTH 个点
    # 因此，验证集的数量保持近似一致即可
    train_count = int(len(X) * 0.8)
    
    X_train, X_val = X_to[:train_count], X_to[train_count:]
    y_train, y_val = y_to[:train_count], y_to[train_count:]
    
    # DataLoader (训练集可以 Shuffle，验证集不需要)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = AttentionBiGRU(
        input_dim=X.shape[2], 
        hidden_dim=config.HIDDEN_DIM, 
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5) # L2 正则化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Start Training on {device}...")
    
    for epoch in range(config.EPOCHS):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)
        
        for bx, by in pbar:
            optimizer.zero_grad()
            
            target_return = by[:, 0].unsqueeze(1)
            target_vol = by[:, 1].unsqueeze(1)
            
            # 前向传播 (Forward)
            # pred_return: 模型现在预测的是收益率
            pred_return, log_var, pred_vol, _ = model(bx)
            
            # 计算损失
            # 1. 收益率预测: Huber Loss + NLL
            huber_loss_val = nn.SmoothL1Loss()(pred_return, target_return)
            nll_loss_val = gaussian_nll_loss(pred_return, log_var, target_return)
            loss_p = huber_loss_val + 0.1 * nll_loss_val
            
            # 2. 方向损失 (优化方向)
            # 由于我们直接预测收益率(变化量)，pred_return 本身就是 Diff
            # target_return 也是 Diff
            loss_dir = direction_loss(pred_return, target_return)
            
            # 3. 波动率损失: MSE (均方误差)
            loss_v = nn.MSELoss()(pred_vol, target_vol)
            
            # 总损失: 
            # 策略调整: 降低回归损失权重，巨幅提升方向/符号损失权重
            # 强迫模型如果不确定数值，至少要猜对正负号，从而消除“永远预测昨日价格”的惰性
            loss = 0.5 * loss_p + 10.0 * loss_dir + loss_v
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                target_return = by[:, 0].unsqueeze(1)
                target_vol = by[:, 1].unsqueeze(1)
                
                pred_return, log_var, pred_vol, _ = model(bx)
                
                huber_loss_val = nn.SmoothL1Loss()(pred_return, target_return)
                nll_loss_val = gaussian_nll_loss(pred_return, log_var, target_return)
                loss_p = huber_loss_val + 0.1 * nll_loss_val
                
                # 验证集方向损失
                loss_dir = direction_loss(pred_return, target_return)
                
                loss_v = nn.MSELoss()(pred_vol, target_vol)
                # 保持一致: 高权重方向惩罚
                loss = 0.5 * loss_p + 10.0 * loss_dir + loss_v
                
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        
        # --- 早停与检查点保存 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{config.MODEL_PATH}")
            print(f"  -> 模型已保存 (新最佳验证损失: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{config.PATIENCE}")
            
        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()

