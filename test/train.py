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
from model import OilPriceTransformer
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
    # [核心修正] 训练目标变更为 Target_Return (对数收益率) 和 Target_Volatility
    # Target_Price 仅用于评估，不参与训练缩放
    target_cols_train = ["Target_Return", "Target_Volatility"] 
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 初始化 Standard Scaler
    scaler = StandardScaler()
    df_features = df[feature_cols]
    df_targets = df[target_cols_train]
    
    # 归一化特征
    data_scaled = scaler.fit_transform(df_features)
    # 归一化目标值
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(df_targets) # 只包含 Return 和 Vol
    
    # 保存 Scalers 以便后续推理使用
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_features.pkl")
    joblib.dump(scaler_target, "models/scaler_targets.pkl")
    joblib.dump(feature_cols, "models/feature_names.pkl")
    
    # 合并数据以创建时间序列
    combined_data = np.hstack([data_scaled, target_scaled])
    
    # create_sequences 需要知道有几个目标列
    X, y = create_sequences(combined_data, config.SEQ_LENGTH, target_cols_train)
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
    input_dim = X_train.shape[2]
    config.INPUT_DIM = input_dim
    print(f"初始化模型: Transformer Encoder (d_model={config.HIDDEN_DIM}, nhead={config.NHEAD if hasattr(config, 'NHEAD') else 4})")
    # 模型定义
    model = OilPriceTransformer(
        input_dim=input_dim, 
        hidden_dim=config.HIDDEN_DIM, 
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        nhead=config.NHEAD if hasattr(config, 'NHEAD') else 4
    ).to(device)
    
    # [优化] 使用 AdamW + ReduceLROnPlateau
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY if hasattr(config, 'WEIGHT_DECAY') else 1e-4) # 增加 Weight Decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
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
            
            # [核心修改] Target 现在是 Log Return 而不是 Price
            target_return = by[:, 0].unsqueeze(1)
            # Volatility 仍保留，但权重降低
            target_vol = by[:, 1].unsqueeze(1)
            
            # 前向传播 (Forward)
            pred_return, log_var, pred_vol, _ = model(bx)
            
            # ========== 损失函数设计 (针对滞后问题优化) ==========
            
            # 1. 基础 MSE 损失
            loss_mse = nn.MSELoss()(pred_return, target_return)
            
            # 2. 方向准确性损失
            loss_dir = direction_loss(pred_return, target_return)
            
            # 3. 波动率损失
            loss_v = nn.MSELoss()(pred_vol, target_vol)
            
            # 4. [关键] 幅度匹配损失 - 惩罚预测幅度过小
            pred_magnitude = torch.abs(pred_return)
            target_magnitude = torch.abs(target_return)
            loss_magnitude = nn.MSELoss()(pred_magnitude, target_magnitude)
            
            # 5. [关键] 非对称敏感度损失 - 更重惩罚"漏掉大变化"
            # 当真实变化大但预测小时，惩罚加重
            underestimate_penalty = torch.relu(target_magnitude - pred_magnitude) * 2.0
            overestimate_penalty = torch.relu(pred_magnitude - target_magnitude) * 0.5
            loss_sensitivity = (underestimate_penalty + overestimate_penalty).mean()
            
            # 6. [新增] 趋势一致性损失 - 惩罚与近期趋势矛盾的预测
            # 计算输入序列中最近5天的平均收益率方向
            recent_returns = bx[:, -5:, 0]  # 假设第0列是 Oil_Close
            recent_trend = (recent_returns[:, -1] - recent_returns[:, 0]).sign().unsqueeze(1)
            pred_sign = pred_return.sign()
            # 如果预测方向与近期趋势一致，损失为0；否则产生小惩罚
            # 注意：这个权重要小，因为趋势反转也是合理的
            trend_conflict = torch.relu(-recent_trend * pred_sign) * 0.1
            loss_trend = trend_conflict.mean()
            
            # 综合损失 (重点提高幅度和敏感度权重)
            loss = (loss_mse 
                    + 2.0 * loss_dir 
                    + 0.1 * loss_v 
                    + 3.0 * loss_magnitude 
                    + 2.0 * loss_sensitivity
                    + 0.5 * loss_trend)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # 显示
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
             for bx, by in val_loader:
                target_return = by[:, 0].unsqueeze(1)
                
                pred_return, log_var, _, _ = model(bx)
                
                loss_mse = nn.MSELoss()(pred_return, target_return)
                loss_dir = direction_loss(pred_return, target_return)
                
                # 验证损失与训练保持一致
                pred_magnitude = torch.abs(pred_return)
                target_magnitude = torch.abs(target_return)
                loss_magnitude = nn.MSELoss()(pred_magnitude, target_magnitude)
                
                underestimate_penalty = torch.relu(target_magnitude - pred_magnitude) * 2.0
                overestimate_penalty = torch.relu(pred_magnitude - target_magnitude) * 0.5
                loss_sensitivity = (underestimate_penalty + overestimate_penalty).mean()
                
                val_loss = loss_mse + 2.0 * loss_dir + 3.0 * loss_magnitude + 2.0 * loss_sensitivity
                
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # [新增] 计算更多诊断指标
        model.eval()
        with torch.no_grad():
            # 在验证集上采样一个批次，计算预测的平均变化幅度
            sample_x, sample_y = next(iter(val_loader))
            sample_pred, _, _, _ = model(sample_x)
            
            # 预测变化的平均绝对值 (检测是否预测"几乎不变")
            pred_change_mag = torch.abs(sample_pred[:, 0]).mean().item()
            target_change_mag = torch.abs(sample_y[:, 0]).mean().item()
            
            # 方向准确率
            pred_sign = torch.sign(sample_pred[:, 0])
            target_sign = torch.sign(sample_y[:, 0])
            direction_acc = (pred_sign == target_sign).float().mean().item()
        
        # 学习率调整 (基于验证Loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f} | "
              f"PredMag={pred_change_mag:.4f} | TgtMag={target_change_mag:.4f} | DirAcc={direction_acc:.2%}", end=" ")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存
            torch.save(model.state_dict(), f"models/{config.MODEL_PATH}")
            print(f"-> [Saved] New Best Model")
        else:
            patience_counter += 1
            print(f"-> [Patience] {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                print(f"Validation loss did not improve for {config.PATIENCE} epochs. Early stopping...")
                break
    
    # [新增] 训练结束后的诊断报告
    print("\n" + "="*60)
    print("训练完成！运行诊断分析...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # 在完整验证集上评估
        all_preds = []
        all_targets = []
        for bx, by in val_loader:
            pred, _, _, _ = model(bx)
            all_preds.append(pred[:, 0].cpu().numpy())
            all_targets.append(by[:, 0].cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # 统计
        pred_mean = np.mean(np.abs(all_preds))
        target_mean = np.mean(np.abs(all_targets))
        pred_std = np.std(all_preds)
        target_std = np.std(all_targets)
        
        # 相关系数
        corr = np.corrcoef(all_preds, all_targets)[0, 1]
        
        # 方向准确率
        dir_acc = np.mean(np.sign(all_preds) == np.sign(all_targets))
        
        print(f"验证集诊断 (归一化的 Log Return):")
        print(f"  预测平均幅度: {pred_mean:.4f} | 真实平均幅度: {target_mean:.4f}")
        print(f"  预测标准差: {pred_std:.4f} | 真实标准差: {target_std:.4f}")
        print(f"  相关系数: {corr:.3f}")
        print(f"  方向准确率: {dir_acc:.2%}")
        
        if pred_mean < target_mean * 0.5:
            print(f"\n⚠️  警告: 预测幅度过小 (仅为真实值的 {pred_mean/target_mean:.1%})，模型可能过于保守！")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    train()
