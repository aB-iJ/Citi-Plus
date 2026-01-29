"""
增强版训练脚本 - 测试更复杂的模型是否能提升性能
"""
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

from config_enhanced import config_enhanced as config
from data_loader import get_processed_data
from model_enhanced import EnhancedAttentionBiGRU
from utils import get_device, set_seed

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

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = nn.BCELoss(reduction='none')(pred, target)
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()

def train():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")
    print(f"\n[增强模型配置]")
    print(f"- 序列长度: {config.SEQ_LENGTH}")
    print(f"- 隐藏维度: {config.HIDDEN_DIM}")
    print(f"- GRU层数: {config.NUM_LAYERS}")
    print(f"- 注意力头数: {config.NUM_HEADS}")
    print(f"- Dropout: {config.DROPOUT}")
    print(f"- 批次大小: {config.BATCH_SIZE}")
    print(f"- 学习率: {config.LEARNING_RATE}")
    
    df = get_processed_data()
    
    stationary_cols = [
        'Dist_SMA_5', 'Dist_SMA_20', 
        'RSI', 'MACD', 
        'ROC_1', 'ROC_3', 'ROC_5', 
        'ROC_Accel_1', 'ROC_Accel_3', 'RSI_Momentum',
        'Bollinger_PctB', 'Bollinger_Width',
        'News_Impact',
        'Oil_Volume'
    ]
    for t in config.TICKERS_FACTORS:
        ret_col = f"{t}_Ret"
        if ret_col in df.columns:
            stationary_cols.append(ret_col)
            
    feature_cols = [c for c in stationary_cols if c in df.columns]
    target_cols = ["Target_Direction"]
    
    print(f"\n特征数量: {len(feature_cols)}")
    
    scaler = StandardScaler()
    border_idx = int(len(df) * 0.8)
    train_df = df.iloc[:border_idx]
    
    scaler.fit(train_df[feature_cols])
    data_scaled = scaler.transform(df[feature_cols])
    target_data = df[target_cols].values
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_features_enhanced.pkl")
    joblib.dump(feature_cols, "models/feature_names_enhanced.pkl")
    
    combined_data = np.hstack([data_scaled, target_data])
    
    X, y = create_sequences(combined_data, config.SEQ_LENGTH, target_cols)
    print(f"总序列数: {X.shape[0]}")
    
    up_count = (y[:, 0] == 1).sum()
    down_count = (y[:, 0] == 0).sum()
    print(f"类别分布: UP={up_count} ({up_count/len(y):.1%}), DOWN={down_count} ({down_count/len(y):.1%})")
    
    X_to = torch.FloatTensor(X).to(device)
    y_to = torch.FloatTensor(y).to(device)
    
    train_count = int(len(X) * 0.8)
    X_train, X_val = X_to[:train_count], X_to[train_count:]
    y_train, y_val = y_to[:train_count], y_to[train_count:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 使用增强模型
    model = EnhancedAttentionBiGRU(
        input_dim=X.shape[2], 
        hidden_dim=config.HIDDEN_DIM, 
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_acc = 0
    patience_counter = 0
    
    print(f"\n开始训练...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)
        
        for bx, by in pbar:
            optimizer.zero_grad()
            
            target_dir = by[:, 0].unsqueeze(1)
            pred_dir, pred_conf, _ = model(bx)
            
            loss = focal_loss(pred_dir, target_dir)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            pred_label = (pred_dir > 0.5).float()
            train_correct += (pred_label == target_dir).sum().item()
            train_total += target_dir.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.2%}"})
        
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                target_dir = by[:, 0].unsqueeze(1)
                pred_dir, pred_conf, _ = model(bx)
                loss = focal_loss(pred_dir, target_dir)
                
                total_val_loss += loss.item()
                pred_label = (pred_dir > 0.5).float()
                val_correct += (pred_label == target_dir).sum().item()
                val_total += target_dir.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} Acc {train_acc:.2%} | Val Loss {avg_val_loss:.4f} Acc {val_acc:.2%} | LR {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{config.MODEL_PATH}")
            print(f"  ✓ 模型已保存 (新最佳验证准确率: {best_val_acc:.2%})")
        else:
            patience_counter += 1
            print(f"  → 未改进. Patience {patience_counter}/{config.PATIENCE}")
            
        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2%}")
    print(f"\n对比基线模型 (53.71%), 增强模型准确率: {best_val_acc:.2%}")
    improvement = (best_val_acc - 0.5371) * 100
    print(f"性能提升: {improvement:+.2f}%")

if __name__ == "__main__":
    train()
