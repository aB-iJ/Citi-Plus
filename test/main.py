import torch
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import config
from model import AttentionBiGRU
from data_loader import get_processed_data
from utils import get_device

def load_environment():
    device = get_device()
    
    # 加载 Scalers
    try:
        scaler_features = joblib.load("models/scaler_features.pkl")
        scaler_targets = joblib.load("models/scaler_targets.pkl")
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
        # 使用 weights_only=True 加载以抑制警告，如果 torch 版本太旧则回退
        model.load_state_dict(torch.load(f"models/{config.MODEL_PATH}", map_location=device))
    except:
         model.load_state_dict(torch.load(f"models/{config.MODEL_PATH}", map_location=device, weights_only=False))
         
    model.to(device)
    model.eval()
    
    return model, scaler_features, scaler_targets, feature_names, device

def evaluate_and_plot_history(days_to_plot=200):
    """
    在近期历史数据上回测模型并绘制详细图表
    """
    print(f"\n正在评估过去 {days_to_plot} 天的模型表现...")
    env = load_environment()
    if not env: return
    model, scaler_f, scaler_t, feature_names, device = env
    
    df = get_processed_data()
    # 我们需要序列。
    # 让我们重构过去 N 天的序列。
    
    # 提取数据
    data_feat = df[feature_names].values
    data_target = df[["Target_Price", "Oil_Close"]].values # Oil Close 是实际值。Target Price 是下一天的值。
    
    # 我们想要使用 T-60..T 来预测 T 时刻的 Target_Price
    
    predictions_price = []
    predictions_upper = []
    predictions_lower = []
    confidence_scores = []
    actual_prices = []
    dates = []
    
    # 遍历最后 N 天
    # 确保我们有足够的历史数据作为序列长度
    start_idx = len(df) - days_to_plot
    if start_idx < config.SEQ_LENGTH:
        start_idx = config.SEQ_LENGTH
        
    indices = range(start_idx, len(df))
    
    print("正在生成预测...")
    with torch.no_grad():
        for i in tqdm(indices):
            # 输入序列: i-SEQ_LEN 到 i
            seq_raw = data_feat[i-config.SEQ_LENGTH : i]
            # 修复警告: 包装在 DataFrame 中
            seq_df = pd.DataFrame(seq_raw, columns=feature_names)
            seq_scaled = scaler_f.transform(seq_df)
            input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
            
            # 预测
            pred_price_scaled, pred_log_var, pred_vol_scaled, _ = model(input_tensor)
            
            # 反归一化价格和波动率
            p_val = pred_price_scaled.cpu().numpy()[0][0]
            v_val = pred_vol_scaled.cpu().numpy()[0][0]
            # 虚拟反归一化
            inv = scaler_t.inverse_transform([[p_val, v_val]])[0]
            final_price = inv[0]
            final_vol = inv[1]
            
            # 置信度
            log_var = pred_log_var.cpu().numpy()[0][0]
            sigma = np.sqrt(np.exp(log_var))
            conf = np.exp(-sigma) # 简化的 0-1 分数
            
            predictions_price.append(final_price)
            predictions_upper.append(final_price + final_vol/2)
            predictions_lower.append(final_price - final_vol/2)
            confidence_scores.append(conf)
            
            # 真实目标 (T+1 时刻的价格，对应于此次预测)
            # 在 dataframe 中，行 i 的 'Target_Price' 就是 T+1 时刻的价格。(检查数据加载器的 shift)
            # data_loader: df['Target_Price'] = df['Oil_Close'].shift(-1)
            # 所以行 i 的 Target_Price 确实是在行 i 做出预测的 ground truth。
            actual_prices.append(df.iloc[i]['Target_Price'])
            dates.append(df.index[i])
            
    # 移除 NaN (如果有) (最后一行可能包含 NaN target)
    valid_idx = [i for i, p in enumerate(actual_prices) if not np.isnan(p)]
    
    # 过滤列表
    dates = [dates[i] for i in valid_idx]
    actual = [actual_prices[i] for i in valid_idx]
    preds = [predictions_price[i] for i in valid_idx]
    upper = [predictions_upper[i] for i in valid_idx]
    lower = [predictions_lower[i] for i in valid_idx]
    confs = [confidence_scores[i] for i in valid_idx]
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. 价格 & 范围
    ax1.plot(dates, actual, label="Actual Oil Price", color="black", linewidth=2)
    ax1.plot(dates, preds, label="AI Predicted Price", color="royalblue", linestyle="--")
    ax1.fill_between(dates, lower, upper, color="royalblue", alpha=0.2, label="Predicted Volatility Range")
    ax1.set_title("Oil Price Prediction vs Actual (Hybrid Transformer-LSTM)", fontsize=14)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. 置信度
    ax2.plot(dates, confs, label="Model Confidence Score", color="green")
    ax2.set_ylabel("Confidence (0-1)")
    ax2.set_xlabel("Date")
    ax2.fill_between(dates, 0, confs, color="green", alpha=0.1)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("prediction_analysis.png", dpi=300)
    print("图表已保存至 prediction_analysis.png")

def explain_model_shap():
    print("\n开始特征重要性分析...")
    env = load_environment()
    if not env: return
    model, scaler_f, scaler_t, feature_names, device = env
    
    # 准备数据
    df = get_processed_data()
    data_feat = df[feature_names].values
    # 修复警告: data_feat 是 numpy 数组，如果 scaler 是在 DataFrame 上拟合的，transform 需要 DataFrame
    # scaler_f.transform(data_feat) causing warning
    df_feat_temp = pd.DataFrame(data_feat, columns=feature_names)
    data_scaled = scaler_f.transform(df_feat_temp)
    
    # 使用基于梯度的显著性 (输入梯度) 代替 SHAP DeepExplainer
    # 原因: DeepExplainer 在较新版本的 PyTorch 中 LayerNorm/LSTM 上会中断。
    # 输入梯度是特征重要性的鲁棒代理。
    
    # 取最近数据的样本 (例如: 最后100天)
    sample_size = 100
    if len(data_scaled) < config.SEQ_LENGTH + sample_size:
        print("数据不足以进行解释。")
        return
        
    # 创建输入张量批次
    inputs = []
    for i in range(len(data_scaled) - sample_size, len(data_scaled)):
        seq = data_scaled[i-config.SEQ_LENGTH : i]
        inputs.append(seq)
    
    input_tensor = torch.FloatTensor(np.array(inputs)).to(device)
    input_tensor.requires_grad = True
    
    # 前向传播
    pred_price, _, _, _ = model(input_tensor)
    
    # 后向传播以获取关于输入的梯度
    # 预测总和是标量，允许反向传播
    pred_price.sum().backward()
    
    # 梯度: (Batch, Seq, Features)
    grads = input_tensor.grad.abs().cpu().numpy()
    
    # 在 Batch 和 Sequence 上取平均以获得全局特征重要性
    # 我们想知道哪个 FEATURE 最重要，无论时间步长如何
    feature_importance = np.mean(grads, axis=(0, 1))
    
    # 归一化到 0-1
    feature_importance = feature_importance / feature_importance.sum()
    
    # 排序
    sorted_idx = np.argsort(feature_importance)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = feature_importance[sorted_idx]
    
    # 绘图
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_vals, color='teal')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel("Relative Importance Score (Gradient-based Impact)")
    plt.title("What drives Oil Prices? (AI Feature Analysis)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("特征重要性图表已保存至 feature_importance.png")

def validate_model_performance():
    """
    在部分训练集和测试集上评估模型，以检查过拟合/欠拟合情况。
    """
    print("\n--- 开始详细模型验证 ---")
    env = load_environment()
    if not env: return
    model, scaler_f, scaler_t, feature_names, device = env
    
    df = get_processed_data()
    
    # 1. 检查数据质量
    print("\n[数据质量检查]")
    print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"总行数: {len(df)}")
    print(f"缺失值: {df.isnull().sum().sum()}")
    print("样例数据 (前2行):")
    print(df[feature_names].head(2))
    
    total_len = len(df)
    train_size = int((total_len - config.SEQ_LENGTH) * 0.8) + config.SEQ_LENGTH
    
    # 定义评估索引
    # 训练评估: 取训练部分的最后300条
    train_eval_start = max(config.SEQ_LENGTH, train_size - 300)
    train_indices = range(train_eval_start, train_size)
    
    # 测试评估: 从 train_size 到结束 (减去可预测步数)
    test_indices = range(train_size, total_len - config.PREDICT_STEPS)
    
    data_feat = df[feature_names].values
    
    def run_inference(indices, label):
        preds = []
        actuals = []
        dates = []
        uppers = []
        lowers = []
        
        print(f"正在对 {label} 集运行推断 ({len(indices)} 样本)...")
        with torch.no_grad():
            for i in tqdm(indices):
                if i < config.SEQ_LENGTH: continue
                
                # 输入: [i-Seq ... i-1]
                seq_raw = data_feat[i-config.SEQ_LENGTH : i]
                # 修复: 包装在 DataFrame 中以消除警告
                seq_df = pd.DataFrame(seq_raw, columns=feature_names)
                seq_scaled = scaler_f.transform(seq_df)
                
                input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                
                pred_p, log_var, _, _ = model(input_tensor)
                
                # 反归一化价格
                p_val = pred_p.cpu().numpy()[0][0]
                final_price = scaler_t.inverse_transform([[p_val, 0]])[0][0] 
                
                # 反归一化不确定性
                start_log_var = log_var.cpu().numpy()[0][0]
                sigma_scaled = np.exp(0.5 * start_log_var)
                price_scale_factor = scaler_t.scale_[0]
                sigma_real = sigma_scaled * price_scale_factor
                
                preds.append(final_price)
                uppers.append(final_price + 1.96 * sigma_real)
                lowers.append(final_price - 1.96 * sigma_real)
                
                # 真实值: row i-1 的 target 是 Price(i)
                actual_val = df.iloc[i-1]['Target_Price']
                actuals.append(actual_val)
                dates.append(df.index[i-1]) # 预测是在 i-1 时刻做出的
                
        return dates, actuals, preds, uppers, lowers

    # 运行
    t_dates, t_act, t_pred, t_up, t_low = run_inference(train_indices, "TRAIN (训练集子集)")
    v_dates, v_act, v_pred, v_up, v_low = run_inference(test_indices, "TEST (测试集)")
    
    # 指标计算
    def get_metrics(act, pred):
        act = np.array(act)
        pred = np.array(pred)
        if len(act) == 0: return 0, 0
        mse = np.mean((act - pred)**2)
        mae = np.mean(np.abs(act - pred))
        return mse, mae
        
    t_mse, t_mae = get_metrics(t_act, t_pred)
    v_mse, v_mae = get_metrics(v_act, v_pred)
    
    print(f"\n[性能指标]")
    print(f"训练集子集 - MSE: {t_mse:.4f}, MAE: {t_mae:.4f}")
    print(f"测试集     - MSE: {v_mse:.4f}, MAE: {v_mae:.4f}")
    
    # 绘图
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 训练集绘图
        ax1.plot(t_dates, t_act, label="Actual (Target)", color='black')
        ax1.plot(t_dates, t_pred, label="Predicted", color='blue', linestyle='--')
        ax1.fill_between(t_dates, t_low, t_up, color='blue', alpha=0.15, label="95% CI")
        ax1.set_title(f"Training Set Fit (Last 300 days) - MAE: {t_mae:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 测试集绘图
        ax2.plot(v_dates, v_act, label="Actual (Target)", color='black')
        ax2.plot(v_dates, v_pred, label="Predicted", color='red', linestyle='--')
        ax2.fill_between(v_dates, v_low, v_up, color='red', alpha=0.15, label="95% CI")
        ax2.set_title(f"Test Set Evaluation (Unseen) - MAE: {v_mae:.2f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("model_validation_comparison.png", dpi=300)
        print("验证对比图已保存至 model_validation_comparison.png")
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    validate_model_performance() # 新的详细验证
    
    # 生成用户请求的 "丰富" 图表 (仅测试集)
    print("\n--- 生成带置信区间的完整预测图 (仅测试集) ---")
    env = load_environment()
    if env:
        model, scaler_f, scaler_t, features, device = env
        df = get_processed_data()
        
        # 计算测试集起点
        total_len = len(df)
        train_size = int((total_len - config.SEQ_LENGTH) * 0.8) + config.SEQ_LENGTH
        
        # 测试集数据
        # 注意: 预测索引 i 需要 [i-Seq : i] 的数据
        # 我们从 train_size 开始
        test_indices = range(train_size, total_len)
        
        all_feat = df[features].values
        
        preds = []
        confidences = [] 
        uppers = []
        lowers = []
        actuals = []
        plot_dates = []
        
        with torch.no_grad():
            for i in tqdm(test_indices):
                # 获取以 i-1 结尾的序列以预测 i
                seq_raw = all_feat[i-config.SEQ_LENGTH : i]
                
                # 修复警告
                seq_df = pd.DataFrame(seq_raw, columns=features)
                seq_scaled = scaler_f.transform(seq_df)
                
                input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                
                pred_p, log_var, _, _ = model(input_tensor)
                
                # 反归一化价格
                p_val = pred_p.cpu().item()
                price = scaler_t.inverse_transform([[p_val, 0]])[0][0]
                
                # 反归一化不确定性
                sigma_scaled = np.exp(0.5 * log_var.cpu().item())
                price_scale_factor = scaler_t.scale_[0] 
                sigma_real = sigma_scaled * price_scale_factor
                
                preds.append(price)
                uppers.append(price + 1.96 * sigma_real)
                lowers.append(price - 1.96 * sigma_real)
                
                # 置信度分数 (启发式: 0-1)
                # 限制 Sigma 计算范围避免过小置信度
                # Sigma 例如 2.0 -> Conf 0.33. Sigma 0.5 -> Conf 0.66
                # 改进缩放: 将合理的波动率映射到概率区间
                # 使用指数衰减使分数看起来更像概率
                conf_score = np.exp(-0.5 * sigma_real) 
                confidences.append(conf_score) 
                
                # 真实值
                # row i-1 的 target 是 Price(i)
                # 如果我们预测时间 'i', 比较的是 'i-1' 的 Target_Price
                try:
                    act = df.iloc[i-1]['Target_Price']
                    actuals.append(act)
                    plot_dates.append(df.index[i-1])
                except:
                    pass

        # 绘图 (双轴或子图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 顶部: 价格
        ax1.plot(plot_dates, actuals, label="Actual Price", color="black", linewidth=1.5)
        ax1.plot(plot_dates, preds, label="AI Prediction", color="royalblue", linestyle="--", linewidth=1.5)
        ax1.fill_between(plot_dates, lowers, uppers, color="royalblue", alpha=0.2, label="95% Confidence Interval")
        ax1.set_title("Oil Price Prediction (Test Set Only) - Attention-BiGRU")
        ax1.set_ylabel("Price (USD)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # 底部: 置信度
        ax2.plot(plot_dates, confidences, label="Model Confidence Score (0-1)", color="green", linewidth=1.5)
        ax2.fill_between(plot_dates, 0, confidences, color="green", alpha=0.1)
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("oil_price_prediction_full.png", dpi=300)
        print("测试集预测图已保存至 oil_price_prediction_full.png")

    explain_model_shap()          # Robust gradient explanation
