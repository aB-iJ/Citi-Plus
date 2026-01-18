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
    # 惩罚方向错误的预测
    # 如果乘积 > 0 (方向正确), ReLU 输出 0 -> 损失为 0
    # 如果乘积 < 0 (方向错误), ReLU 输出 -Product -> 产生正的损失值
    loss = torch.relu(-1.0 * pred_diff * actual_diff)
    return loss.mean()

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
    
    feature_cols = [c for c in df.columns if "Target_" not in c]
    target_cols = ["Target_Price", "Target_Volatility"]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 初始化 Standard Scaler
    scaler = StandardScaler()
    df_features = df[feature_cols]
    df_targets = df[target_cols]
    
    # 归一化特征
    data_scaled = scaler.fit_transform(df_features)
    # 归一化目标值
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(df_targets)
    
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
    
    # 划分 训练集/验证集 (按时间序列切分，严禁 Shuffle!)
    # 保留最后 20% 作为验证/测试
    train_size = int(len(X) * 0.8)
    X_train, X_val = X_to[:train_size], X_to[train_size:]
    y_train, y_val = y_to[:train_size], y_to[train_size:]
    
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
            
            target_price = by[:, 0].unsqueeze(1)
            target_vol = by[:, 1].unsqueeze(1)
            
            # 前向传播 (Forward)
            # 解包额外的 Attention 输出 (这里忽略，仅用于可视化)
            pred_price, log_var, pred_vol, _ = model(bx)
            
            # 计算损失
            # 1. 价格预测: Huber Loss (鲁棒回归) + NLL (不确定性) 
            huber_loss_val = nn.SmoothL1Loss()(pred_price, target_price)
            nll_loss_val = gaussian_nll_loss(pred_price, log_var, target_price)
            loss_p = huber_loss_val + 0.1 * nll_loss_val
            
            # 2. 方向损失 (优化方向)
            # 假设第0列是价格 (标准归一化后)
            # bx: (Batch, Seq, Features) -> 取序列最后一步, 特征0
            current_price_input = bx[:, -1, 0].unsqueeze(1)
            pred_diff = pred_price - current_price_input
            actual_diff = target_price - current_price_input
            loss_dir = direction_loss(pred_diff, actual_diff)
            
            # 3. 波动率损失: MSE (均方误差)
            loss_v = nn.MSELoss()(pred_vol, target_vol)
            
            # 总损失: 加大方向损失的权重以强制模型学习趋势
            loss = loss_p + 1.0 * loss_dir + loss_v
            
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
                target_price = by[:, 0].unsqueeze(1)
                target_vol = by[:, 1].unsqueeze(1)
                
                pred_price, log_var, pred_vol, _ = model(bx)
                
                huber_loss_val = nn.SmoothL1Loss()(pred_price, target_price)
                nll_loss_val = gaussian_nll_loss(pred_price, log_var, target_price)
                loss_p = huber_loss_val + 0.1 * nll_loss_val
                
                # 验证集方向损失
                current_price_input = bx[:, -1, 0].unsqueeze(1)
                pred_diff = pred_price - current_price_input
                actual_diff = target_price - current_price_input
                loss_dir = direction_loss(pred_diff, actual_diff)
                
                loss_v = nn.MSELoss()(pred_vol, target_vol)
                loss = loss_p + 1.0 * loss_dir + loss_v
                
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
