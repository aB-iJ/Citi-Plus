import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import config
from model import AttentionBiGRU
from data_loader import get_processed_data
from utils import get_device

def load_environment():
    device = get_device()
    
    # 加载 Scalers
    try:
        scaler_features = joblib.load("models/scaler_features.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
    except:
        print("未找到 Scaler 文件。请先运行 train.py。")
        return None
    
    # 加载模型
    input_dim = len(feature_names)
    
    model = AttentionBiGRU(
        input_dim=input_dim, 
        hidden_dim=config.HIDDEN_DIM, 
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    try:
        model.load_state_dict(torch.load(f"models/{config.MODEL_PATH}", map_location=device))
    except:
         model.load_state_dict(torch.load(f"models/{config.MODEL_PATH}", map_location=device, weights_only=False))
         
    model.to(device)
    model.eval()
    
    return model, scaler_features, feature_names, device

def evaluate_classification_model():
    """
    评估纯分类模型的涨跌预测性能
    """
    print("\n--- 开始分类模型评估 ---")
    env = load_environment()
    if not env: return
    model, scaler_f, feature_names, device = env
    
    df = get_processed_data()
    
    print(f"\n[数据信息]")
    print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"总行数: {len(df)}")
    
    total_len = len(df)
    train_size = int((total_len - config.SEQ_LENGTH) * 0.8) + config.SEQ_LENGTH
    
    # 测试集索引
    test_indices = range(train_size, total_len - config.PREDICT_STEPS)
    
    data_feat = df[feature_names].values
    
    # 推断
    pred_probs = []
    pred_labels = []
    actual_labels = []
    confidences = []
    prices = []
    dates = []
    
    print(f"正在对测试集运行推断 ({len(list(test_indices))} 样本)...")
    with torch.no_grad():
        for i in tqdm(test_indices):
            if i < config.SEQ_LENGTH: continue
            target_idx = i - 1 + config.PREDICT_STEPS
            if target_idx >= len(df): continue
            
            seq_raw = data_feat[i-config.SEQ_LENGTH : i]
            seq_df = pd.DataFrame(seq_raw, columns=feature_names)
            seq_scaled = scaler_f.transform(seq_df)
            
            input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
            
            pred_dir, pred_conf, _ = model(input_tensor)
            
            prob = pred_dir.cpu().item()
            conf = pred_conf.cpu().item()
            label = 1 if prob > 0.5 else 0
            
            pred_probs.append(prob)
            pred_labels.append(label)
            confidences.append(conf)
            
            # 真实涨跌
            actual = df.iloc[target_idx]['Target_Direction']
            actual_labels.append(int(actual))
            
            # 实际价格 (用于绘图)
            prices.append(df.iloc[target_idx]['Oil_Close'])
            dates.append(df.index[target_idx])
    
    # 计算指标
    acc = accuracy_score(actual_labels, pred_labels)
    prec = precision_score(actual_labels, pred_labels, zero_division=0)
    rec = recall_score(actual_labels, pred_labels, zero_division=0)
    f1 = f1_score(actual_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(actual_labels, pred_labels)
    
    print(f"\n[分类性能指标]")
    print(f"准确率 (Accuracy): {acc:.2%}")
    print(f"精确率 (Precision): {prec:.2%}")
    print(f"召回率 (Recall): {rec:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"\n混淆矩阵:")
    print(f"             预测跌  预测涨")
    print(f"实际跌       {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"实际涨       {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    return dates, prices, pred_probs, pred_labels, actual_labels, confidences

def plot_classification_results(days_to_show=90):
    """
    绘制涨跌预测结果的可视化图表 - 更直观的版本
    """
    print(f"\n--- 生成涨跌预测可视化图表 (最近 {days_to_show} 天) ---")
    result = evaluate_classification_model()
    if not result: return
    
    dates, prices, pred_probs, pred_labels, actual_labels, confidences = result
    
    # 只取最后 N 天
    n = min(days_to_show, len(dates))
    dates = dates[-n:]
    prices = prices[-n:]
    pred_probs = pred_probs[-n:]
    pred_labels = pred_labels[-n:]
    actual_labels = actual_labels[-n:]
    confidences = confidences[-n:]
    
    # 计算正确/错误预测
    correct = [1 if p == a else 0 for p, a in zip(pred_labels, actual_labels)]
    
    # 计算实际涨跌幅度
    actual_changes = []
    for i in range(1, len(prices)):
        change_pct = (prices[i] - prices[i-1]) / prices[i-1] * 100
        actual_changes.append(change_pct)
    actual_changes.insert(0, 0)  # 第一天没有变化
    
    # 创建更直观的4格子图
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # --- 1. 主图: 价格走势 + 背景色区分预测结果 ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # 绘制价格线
    ax1.plot(dates, prices, color='black', linewidth=2, label='实际油价', zorder=3)
    
    # 用背景色标记预测正确/错误的区域
    for i in range(len(dates)-1):
        if correct[i]:
            color = 'lightgreen' if pred_labels[i] == 1 else 'lightcyan'
        else:
            color = 'lightcoral' if pred_labels[i] == 1 else 'lightyellow'
        ax1.axvspan(dates[i], dates[i+1], alpha=0.3, color=color, zorder=1)
    
    # 在价格点上叠加小标记
    for i in range(len(dates)):
        if correct[i]:
            marker = '▲' if pred_labels[i] == 1 else '▼'
            color = 'darkgreen'
        else:
            marker = '▲' if pred_labels[i] == 1 else '▼'
            color = 'darkred'
        ax1.text(dates[i], prices[i], marker, fontsize=8, ha='center', 
                va='center', color=color, alpha=0.7)
    
    ax1.set_title('油价走势 + AI预测 (绿色背景=预测正确, 红/黄色背景=预测错误)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('油价 (USD)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # --- 2. 预测 vs 实际的对比条形图 ---
    ax2 = fig.add_subplot(gs[1, :])
    
    x_pos = range(len(dates))
    width = 0.4
    
    # 实际涨跌
    colors_actual = ['green' if a == 1 else 'red' for a in actual_labels]
    ax2.bar([x - width/2 for x in x_pos], 
            [1 if a == 1 else -1 for a in actual_labels], 
            width, label='实际涨跌', color=colors_actual, alpha=0.6)
    
    # 预测涨跌
    colors_pred = ['blue' if p == 1 else 'orange' for p in pred_labels]
    ax2.bar([x + width/2 for x in x_pos], 
            [0.8 if p == 1 else -0.8 for p in pred_labels], 
            width, label='AI预测', color=colors_pred, alpha=0.6)
    
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('涨跌方向')
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['跌', '持平', '涨'])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_title('实际涨跌 vs AI预测对比 (柱高=方向强度)', fontsize=11)
    
    # --- 3. 左下: 涨跌幅度 + 预测准确性 ---
    ax3 = fig.add_subplot(gs[2, 0])
    
    colors_change = ['green' if c else 'red' for c in correct]
    ax3.bar(dates, actual_changes, color=colors_change, alpha=0.7, width=1)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.set_ylabel('实际涨跌幅 (%)')
    ax3.set_title('日涨跌幅 (绿=预测对, 红=预测错)', fontsize=10)
    ax3.grid(True, alpha=0.2, axis='y')
    
    # --- 4. 右下: 预测概率 + 真实标签 ---
    ax4 = fig.add_subplot(gs[2, 1])
    
    # 绘制预测概率
    ax4.fill_between(dates, 0.5, pred_probs, where=[p > 0.5 for p in pred_probs], 
                     color='green', alpha=0.3, label='预测看涨区')
    ax4.fill_between(dates, pred_probs, 0.5, where=[p <= 0.5 for p in pred_probs], 
                     color='red', alpha=0.3, label='预测看跌区')
    ax4.plot(dates, pred_probs, color='blue', linewidth=1.5, label='看涨概率')
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    
    # 叠加实际涨跌的散点
    ax4.scatter([dates[i] for i in range(len(dates)) if actual_labels[i] == 1], 
                [0.9] * sum(actual_labels), color='green', marker='^', s=30, 
                alpha=0.5, label='实际涨')
    ax4.scatter([dates[i] for i in range(len(dates)) if actual_labels[i] == 0], 
                [0.1] * (len(actual_labels) - sum(actual_labels)), color='red', 
                marker='v', s=30, alpha=0.5, label='实际跌')
    
    ax4.set_ylabel('概率')
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper left', fontsize=7)
    ax4.grid(True, alpha=0.2)
    ax4.set_title('AI预测概率 vs 实际结果', fontsize=10)
    
    # --- 5. 左下2: 滚动准确率 ---
    ax5 = fig.add_subplot(gs[3, 0])
    
    window = 20  # 20天滚动窗口
    rolling_acc = []
    for i in range(len(correct)):
        if i < window:
            rolling_acc.append(sum(correct[:i+1]) / (i+1))
        else:
            rolling_acc.append(sum(correct[i-window+1:i+1]) / window)
    
    ax5.plot(dates, rolling_acc, color='orange', linewidth=2, label=f'{window}天滚动准确率')
    ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='随机基准(50%)')
    ax5.fill_between(dates, 0.5, rolling_acc, where=[r > 0.5 for r in rolling_acc], 
                     color='green', alpha=0.2)
    ax5.fill_between(dates, rolling_acc, 0.5, where=[r <= 0.5 for r in rolling_acc], 
                     color='red', alpha=0.2)
    ax5.set_ylabel('准确率')
    ax5.set_ylim(0.3, 0.7)
    ax5.legend(loc='lower left', fontsize=8)
    ax5.grid(True, alpha=0.2)
    ax5.set_title('模型表现稳定性分析', fontsize=10)
    ax5.set_xlabel('日期')
    
    # --- 6. 右下2: 混淆矩阵可视化 ---
    ax6 = fig.add_subplot(gs[3, 1])
    
    cm = confusion_matrix(actual_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax6.imshow(cm_normalized, cmap='RdYlGn', vmin=0, vmax=1)
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['预测跌', '预测涨'])
    ax6.set_yticklabels(['实际跌', '实际涨'])
    
    # 添加数值标注
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax6.set_title('混淆矩阵 (准确率分布)', fontsize=10)
    plt.colorbar(im, ax=ax6, label='归一化比例')
    
    plt.savefig("direction_prediction_analysis.png", dpi=300, bbox_inches='tight')
    print("涨跌预测分析图已保存至 direction_prediction_analysis.png")
    
    # --- 额外: 按置信度分组的准确率 ---
    print("\n[按置信度分组的准确率]")
    conf_bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in conf_bins:
        mask = [(low <= c < high) for c in confidences]
        if sum(mask) > 0:
            bin_correct = [correct[i] for i in range(len(correct)) if mask[i]]
            bin_acc = sum(bin_correct) / len(bin_correct)
            print(f"  置信度 [{low:.1f}, {high:.1f}): {len(bin_correct)} 样本, 准确率 {bin_acc:.2%}")

def explain_model_features():
    """
    使用梯度分析特征重要性
    """
    print("\n--- 开始特征重要性分析 ---")
    env = load_environment()
    if not env: return
    model, scaler_f, feature_names, device = env
    
    df = get_processed_data()
    data_feat = df[feature_names].values
    df_feat_temp = pd.DataFrame(data_feat, columns=feature_names)
    data_scaled = scaler_f.transform(df_feat_temp)
    
    sample_size = 100
    if len(data_scaled) < config.SEQ_LENGTH + sample_size:
        print("数据不足以进行解释。")
        return
    
    inputs = []
    for i in range(len(data_scaled) - sample_size, len(data_scaled)):
        seq = data_scaled[i-config.SEQ_LENGTH : i]
        inputs.append(seq)
    
    input_tensor = torch.FloatTensor(np.array(inputs)).to(device)
    input_tensor.requires_grad = True
    
    # 前向传播
    pred_dir, _, _ = model(input_tensor)
    
    # 后向传播
    pred_dir.sum().backward()
    
    grads = input_tensor.grad.abs().cpu().numpy()
    feature_importance = np.mean(grads, axis=(0, 1))
    feature_importance = feature_importance / feature_importance.sum()
    
    sorted_idx = np.argsort(feature_importance)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = feature_importance[sorted_idx]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_vals, color='teal')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel("Relative Importance Score (Gradient-based)")
    plt.title("Feature Importance for Direction Prediction")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("特征重要性图表已保存至 feature_importance.png")

if __name__ == "__main__":
    plot_classification_results(days_to_show=90)
    explain_model_features()
