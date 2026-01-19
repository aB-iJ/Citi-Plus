import torch
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import config
from model import OilPriceTransformer
from data_loader import get_processed_data
from utils import get_device
from news_agent import NewsCrawler, DeepSeekAnalyzer 

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
    model_path = f"models/{config.MODEL_PATH}"
    if os.path.exists(model_path):
        mod_time = os.path.getmtime(model_path)
        import datetime
        ts = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Loading Model: {model_path} (Last Modified: {ts})")
    else:
        print(f"ERROR: Model file {model_path} not found!")
        return None
        
    input_dim = len(feature_names)
    
    model = OilPriceTransformer(
        input_dim=input_dim, 
        hidden_dim=config.HIDDEN_DIM, 
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        nhead=config.NHEAD if hasattr(config, 'NHEAD') else 4
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
    data_target = df[["Target_Price", "Oil_Close"]].values # Oil_Close 是实际收盘价。Target_Price 是下一天的目标价格。

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
            
            # 反归一化 - 这里的 inv[0] 是预测的对数收益率 (Log Return)，不是价格
            p_val = pred_price_scaled.cpu().numpy()[0][0]
            v_val = pred_vol_scaled.cpu().numpy()[0][0]
            
            # 使用 scaler_targets 进行反变换 (恢复到原始量级)
            inv = scaler_t.inverse_transform([[p_val, v_val]])[0]
            pred_log_return = inv[0]
            pred_volatility = inv[1]
            
            # [核心修正] 从收益率还原价格
            # 模型使用的是直到 i-1 的数据序列进行预测
            # 基准价格是输入序列最后一个时间点 (i-1) 的收盘价
            last_close_price = df.iloc[i-1]['Oil_Close']
            
            # Price(T+1) = Price(T) * exp(Log_Return)
            final_price = last_close_price * np.exp(pred_log_return)
            
            # 波动率也是相对的，如果需要画图，直接用即可
            final_vol = pred_volatility
            
            # 置信度
            log_var = pred_log_var.cpu().numpy()[0][0]
            sigma = np.sqrt(np.exp(log_var))
            conf = np.exp(-sigma) # 简化的 0-1 分数
            
            predictions_price.append(final_price)
            predictions_upper.append(final_price + final_vol/2)
            predictions_lower.append(final_price - final_vol/2)
            confidence_scores.append(conf)
            
            # 真实目标 (预测对应的那一天 i)
            # 这里的 i 是序列之后的一天，也就是我们要预测的那一天
            # 注意: df.iloc[i]['Target_Price'] 是 i+1 天的价格，我们预测的是 i
            if i < len(df):
                actual_prices.append(df.iloc[i]['Oil_Close'])
                dates.append(df.index[i])
            else:
                # 越界保护
                pass
            
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
    ax1.plot(dates, actual, label="Actual Oil Price (真实油价)", color="black", linewidth=2)
    ax1.plot(dates, preds, label="AI Predicted Price (AI预测油价)", color="royalblue", linestyle="--")
    ax1.fill_between(dates, lower, upper, color="royalblue", alpha=0.2, label="Predicted Context (预测置信区间)")
    ax1.set_title("Oil Price Prediction vs Actual (Hybrid Transformer-LSTM)", fontsize=14)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. 置信度
    ax2.plot(dates, confs, label="Model Confidence Score (模型置信度)", color="green")
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
                

                pred_return_scaled, log_var, _, _ = model(input_tensor)
                
                # [关键修正] 反归一化逻辑适配 "Log Return" 目标
                
                # 1. 反归一化预测值 (得到真实的 Log Return)
                pred_ret_val = pred_return_scaled.cpu().numpy()[0][0]
                # 注意: 我们现在的 Target Scaler 拟合的是 [Log_Return, Volatility]
                # inverse_transform 会返回 [Log_Return_Real, Vol_Real]
                real_log_return = scaler_t.inverse_transform([[pred_ret_val, 0]])[0][0]
                
                # 2. 还原为绝对价格
                # Price(t) = Price(t-1) * exp(Log_Return)
                # 获取当天的收盘价 (作为基准) - 也就是 input sequence 的最后一个点的收盘价
                # 注意 seq_raw 是原始特征值，我们需要找到 'Oil_Close' 所在的列
                if 'Oil_Close' in feature_names:
                     close_idx = list(feature_names).index('Oil_Close')
                     last_close_price = seq_raw[-1, close_idx]
                else:
                     # Fallback (不应该发生)
                     last_close_price = 1.0 
                     
                final_price = last_close_price * np.exp(real_log_return)
                
                # 3. 处理不确定性 (简化处理，假设 sigma 是针对 return 的)
                start_log_var = log_var.cpu().numpy()[0][0]
                sigma_scaled = np.exp(0.5 * start_log_var)
                return_scale_factor = scaler_t.scale_[0]
                sigma_return = sigma_scaled * return_scale_factor
                
                # 价格区间的近似: Price * exp(Return +/- 1.96*Sigma)
                upper_price = last_close_price * np.exp(real_log_return + 1.96 * sigma_return)
                lower_price = last_close_price * np.exp(real_log_return - 1.96 * sigma_return)
                
                preds.append(final_price)
                uppers.append(upper_price)
                lowers.append(lower_price)
                
                # 真实值
                actual_val = df.iloc[i-1]['Target_Price']
                actuals.append(actual_val)
                dates.append(df.index[i-1])
                
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

def predict_tomorrow(api_key=None):
    """
    使用实时新闻分析预测明日油价
    """
    print("\n=== 开始实时推理 (Live Inference) ===")
    
    # 1. 加载环境
    env = load_environment()
    if not env: 
        print("环境加载失败")
        return
    model, scaler_f, scaler_t, feature_names, device = env
    
    # 2. 获取最新数据序列
    # 注意：这里我们拿到的 df 包含了直到最近一个交易日的数据
    df = get_processed_data()
    
    # 只需要最后 SEQ_LENGTH 天的数据来预测明天
    if len(df) < config.SEQ_LENGTH:
        print("数据不足！")
        return

    last_sequence_df = df.iloc[-config.SEQ_LENGTH:].copy()
    
    # 3. 获取实时新闻情绪分数 (替换掉原来的 VIX 代理分数)
    print("\n正在获取今日实时新闻...")
    print(f"DEBUG: API Key present: {bool(api_key)}")
    try:
        crawler = NewsCrawler()
        news_dict = crawler.fetch_investing_com_news()
        
        # 将按日期分组的新闻字典展平为列表，供 API 分析
        all_headlines = []
        for date_str, titles in news_dict.items():
            all_headlines.extend(titles)
        
        if all_headlines and api_key:
            print(f"获取到 {len(all_headlines)} 条新闻 (覆盖 {len(news_dict)} 天)，正在调用 DeepSeek 进行情感分析...")
            analyzer = DeepSeekAnalyzer(api_key=api_key)
            ai_score = analyzer.analyze_sentiment(all_headlines)
            print(f"DeepSeek AI 评分结果: {ai_score} (-1 极空 ~ 1 极多)")
            
            # 关键步骤：修改输入特征中的 News_Impact
            # 我们只修改序列中最后一天 (Today) 的因子值，假设新闻影响是即时的
            if 'News_Impact' in feature_names:
                # 定位到 News_Impact 列
                last_sequence_df.iloc[-1, last_sequence_df.columns.get_loc('News_Impact')] = ai_score
                print("已利用 AI 舆情指数更新模型输入")
            else:
                print("警告: 训练特征中未找到 'News_Impact'，无法注入 AI 因子")
        else:
            if not all_headlines:
                print("未抓取到任何新闻。")
            if not api_key:
                print("未检测到 API Key (api_key is None/Empty)。")
            print("将使用默认计算的代理指标。")
            default_score = last_sequence_df.iloc[-1]['News_Impact']
            print(f"默认 VIX 代理指标得分: {default_score}")

    except Exception as e:
        print(f"新闻模块出错，回退到默认数据: {e}")

    # 4. 预处理 & 推理
    print("正在运行神经网络推理...")
    try:
        # 只取特征列
        seq_feat = last_sequence_df[feature_names]
        
        # 缩放
        seq_scaled = scaler_f.transform(seq_feat)
        
        # 转换为 Tensor
        input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            pred_return_scaled, log_var, _, _ = model(input_tensor)
            
            # [核心修正] 模型预测的是对数收益率 (Log Return)，不是价格！
            # 需要: 1. 反归一化得到真实 Return  2. 用昨收 * exp(Return) 还原价格
            ret_val = pred_return_scaled.cpu().item()
            
            # 反归一化 Return (scaler_t 拟合的是 [Target_Return, Target_Volatility])
            real_return = scaler_t.inverse_transform([[ret_val, 0]])[0][0]
            
            # 获取昨日收盘价 (序列最后一天的 Oil_Close)
            if 'Oil_Close' in feature_names:
                last_close = last_sequence_df.iloc[-1]['Oil_Close']
            else:
                last_close = last_sequence_df.iloc[-1, 0]  # fallback
            
            # 还原预测价格: P_tomorrow = P_today * exp(predicted_return)
            price = last_close * np.exp(real_return)
            
            # 不确定性 (针对 Return 的标准差)
            sigma_scaled = np.exp(0.5 * log_var.cpu().item())
            ret_scale_factor = scaler_t.scale_[0]  # Return 的缩放因子
            sigma_ret = sigma_scaled * ret_scale_factor
            
            # 价格区间 (基于 Return 的置信区间转换为价格)
            price_upper = last_close * np.exp(real_return + 1.96 * sigma_ret)
            price_lower = last_close * np.exp(real_return - 1.96 * sigma_ret)
            
            # 置信度
            conf_score = max(0.1, np.exp(-2.0 * abs(sigma_ret)))

        print("\n" + "="*50)
        print(f"  🛢️  预测结果 (Prediction for Next Trading Day)")
        print("="*50)
        print(f"  昨日收盘价: ${last_close:.2f}")
        print(f"  预测收益率: {real_return*100:.2f}%")
        print(f"  预测价格: ${price:.2f}")
        print(f"  置信区间: [${price_lower:.2f}, ${price_upper:.2f}]")
        print(f"  模型置信度: {conf_score:.1%}")
        
        last_date = last_sequence_df.index[-1].strftime('%Y-%m-%d')
        print(f"  (基于截止至 {last_date} 的数据)")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"推理过程出错: {e}")

if __name__ == "__main__":
    # 模式选择
    # 1. 验证模式: 回测历史，生成图表
    validate_model_performance() 
    
    # 2. 也是验证模式: 生成完整测试集图表
    print("\n--- 生成带置信区间的完整预测图 (AI 增强版) ---")
    
    # 获取 API Key (请确保您已设置 DeepSeek_API 环境变量，或在此处硬编码)
    DEEPSEEK_API_KEY = os.getenv("DeepSeek_API") 
    # DEEPSEEK_API_KEY = "sk-xxxxxxxx" # 您的 Key
    
    # 1. 预先爬取新闻 (如果提供了 Key)
    news_db = {}
    
    if DEEPSEEK_API_KEY:
        print("正在检查并补全过去90天的新闻数据 (DuckDuckGo Search)...")
        try:
            crawler = NewsCrawler()
            # 智能补全: 自动检查本地是否有缺失的日期并联网抓取
            news_db = crawler.crawl_last_n_days(n=90)
            print(f"新闻库最终状态: 包含 {len(news_db)} 天的数据")
            
        except Exception as e:
            print(f"爬虫初始化/运行失败: {e}")
            # 降级: 尝试读取本地缓存
            if os.path.exists("crawled_news.json"):
                try:
                    with open("crawled_news.json", "r", encoding='utf-8') as f:
                        news_db = json.load(f)
                except: pass
    else:
        print("未检测到 API Key，将跳过在线更新，仅尝试读取本地历史新闻...")
        if os.path.exists("crawled_news.json"):
            try:
                with open("crawled_news.json", "r", encoding='utf-8') as f:
                    news_db = json.load(f)
            except: pass

    env = load_environment()
    if env:
        model, scaler_f, scaler_t, features, device = env
        df = get_processed_data()
        
        # [核心修正] 统一推理条件：测试集和 Full 图使用相同的数据
        # 之前的问题：测试集用 Oracle News，Full 图用 VIX 代理，导致表现不一致
        # 现在：两者都使用原始 get_processed_data() 的数据（包含 Oracle）
        # 这样可以公平对比。如果用户有真正的 DeepSeek 新闻分析，AI Model 会用那个。
        
        # 注意：这里不再降级 News_Impact，保持原始数据
        # df["News_Impact"] = vix_proxy...  <- 移除这段代码
             
        # 重新提取特征矩阵
        all_feat = df[features].values
        
        # 计算测试集起点 (为了演示效果，我们重点关注最近 90 天的数据)
        total_len = len(df)
        plot_days = 90 # 扩大一点范围
        # 确保不越界
        start_idx = max(config.SEQ_LENGTH, total_len - plot_days)
        test_indices = range(start_idx, total_len)
        
        preds = []
        preds_ai = [] # 存储 AI 增强后的预测
        confidences = [] 
        uppers = []
        lowers = []
        actuals = []
        plot_dates = []
        
        # AI 分析器实例
        analyzer = None
        if DEEPSEEK_API_KEY:
            analyzer = DeepSeekAnalyzer(api_key=DEEPSEEK_API_KEY)
        
        print(f"开始推理最近 {len(test_indices)} 天的数据 (Base vs AI)...")
        
        # 预先查找 News_Impact 在特征中的列索引
        news_feat_idx = -1
        if 'News_Impact' in features:
            news_feat_idx = list(features).index('News_Impact')

        with torch.no_grad():
            for i in tqdm(test_indices):
                # [核心修正] 预测对齐问题
                # 序列: df[i-SEQ_LENGTH : i]  -> 预测目标: df[i] 的价格
                # 序列最后一天是 df[i-1]，我们用它预测下一天 df[i]
                
                current_date = df.index[i-1]
                date_str = current_date.strftime('%Y-%m-%d')
                
                seq_raw = all_feat[i-config.SEQ_LENGTH : i].copy()
                
                # --- 分支 A: 标准预测 (使用弱化的 VIX 代理) ---
                seq_df = pd.DataFrame(seq_raw, columns=features)
                seq_scaled = scaler_f.transform(seq_df)
                input_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                
                # Model 输出 Return
                pred_ret, log_var, _, _ = model(input_tensor)
                
                # 1. 还原 Price (Base)
                p_ret_val = pred_ret.cpu().item()
                real_ret = scaler_t.inverse_transform([[p_ret_val, 0]])[0][0]
                
                # 获取昨收 (序列最后一天，即 df[i-1])
                if 'Oil_Close' in features:
                     last_close_price = seq_raw[-1, list(features).index('Oil_Close')]
                else: 
                     last_close_price = df.iloc[i-1]['Oil_Close']
                
                # 预测今天 (df[i]) 的价格
                price = last_close_price * np.exp(real_ret)
                
                # 不确定性 (针对 Return)
                sigma_scaled = np.exp(0.5 * log_var.cpu().item())
                ret_scale_factor = scaler_t.scale_[0] 
                sigma_ret = sigma_scaled * ret_scale_factor
                
                preds.append(price)
                # 价格区间
                uppers.append(last_close_price * np.exp(real_ret + 1.96 * sigma_ret))
                lowers.append(last_close_price * np.exp(real_ret - 1.96 * sigma_ret))
                
                conf_score = np.exp(-0.5 * sigma_ret) # 简化
                confidences.append(conf_score) 
                
                # --- 分支 B: AI 增强预测 (注入真实历史新闻) ---
                price_ai = price # 默认
                
                # 只有在有新闻且找到了特征列时才进行增强
                if news_feat_idx >= 0 and date_str in news_db:
                    # 获取该日新闻
                    daily_news = news_db[date_str]
                    
                    if analyzer:
                         # 缓存逻辑
                         if not hasattr(analyzer, 'cache'): analyzer.cache = {}
                         if date_str in analyzer.cache:
                             ai_score = analyzer.cache[date_str]
                         else:
                             if len(daily_news) > 0:
                                 # 简单限流: 如果是 DuckDuckGo 得到的空新闻，不调用
                                 ai_score = analyzer.analyze_sentiment(daily_news)
                             else:
                                 ai_score = 0
                             analyzer.cache[date_str] = ai_score
                    else:
                        ai_score = 0
                    
                    # 构造新的序列用于 AI 推理
                    seq_ai = seq_raw.copy()
                    seq_ai[-1, news_feat_idx] = ai_score 
                    
                    # 重新缩放 & 推理
                    seq_ai_df = pd.DataFrame(seq_ai, columns=features)
                    seq_ai_scaled = scaler_f.transform(seq_ai_df)
                    input_tensor_ai = torch.FloatTensor(seq_ai_scaled).unsqueeze(0).to(device)
                    
                    pred_ret_ai, _, _, _ = model(input_tensor_ai)
                    
                    p_ret_val_ai = pred_ret_ai.cpu().item()
                    real_ret_ai = scaler_t.inverse_transform([[p_ret_val_ai, 0]])[0][0]
                    
                    # 还原价格
                    price_ai = last_close_price * np.exp(real_ret_ai)
                
                preds_ai.append(price_ai)
                
                # [核心修正] 真实值对齐
                # 我们预测的是 df[i] 那天的价格，所以真实值就是 df.iloc[i]['Oil_Close']
                try:
                    actual_price = df.iloc[i]['Oil_Close']
                    actuals.append(actual_price)
                    plot_dates.append(df.index[i])  # 日期也应该是预测目标日 df[i]
                except:
                    pass

        # 绘图 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # [调试] 输出一些统计信息
        print(f"\n[调试信息]")
        print(f"预测数据点数: {len(preds)}")
        print(f"真实数据点数: {len(actuals)}")
        print(f"预测价格范围: ${min(preds):.2f} - ${max(preds):.2f}")
        print(f"真实价格范围: ${min(actuals):.2f} - ${max(actuals):.2f}")
        print(f"平均预测误差: ${np.mean(np.abs(np.array(preds) - np.array(actuals))):.2f}")
        
        # 计算相关系数
        corr = np.corrcoef(preds[:len(actuals)], actuals)[0, 1]
        print(f"预测与真实的相关系数: {corr:.3f}")
        
        # 顶部: 价格对比
        # [修改] 优化绘图样式以解决遮挡问题
        # Base Model: 灰色粗实线，半透明背景
        ax1.plot(plot_dates, preds, label="Base Model (VIX Proxy)", color="gray", linewidth=4, alpha=0.4)
        
        # AI Model: 蓝色细线+点状，叠加在上层
        # 仅当 AI 预测与普通预测不同时才会有明显的视觉差异
        ax1.plot(plot_dates, preds_ai, label="AI-Enhanced Prediction (Real News)", color="royalblue", linewidth=1.5, linestyle="-.")
        
        ax1.fill_between(plot_dates, lowers, uppers, color="royalblue", alpha=0.15, label="95% Confidence Interval")
        ax1.set_title(f"Oil Price Prediction: AI News vs VIX Proxy (Last {len(plot_dates)} Days)")
        ax1.set_ylabel("Price (USD)")
        
        # 强制把真实价格画在最最上层，黑色细实线
        ax1.plot(plot_dates, actuals, label="Actual Price", color="black", linewidth=1, alpha=0.9, zorder=10)
        
        # [诊断绘图] 绘制 Shift(-1) 的真实价格曲线 (Yesterday's Price)
        # 如果预测线与这条线重合，说明模型退化为 Trivial Identity (Persistence Model)
        # 用虚线绘制
        shifted_actuals = [0] + actuals[:-1]
        if len(shifted_actuals) == len(plot_dates):
             ax1.plot(plot_dates, shifted_actuals, label="Persistence Baseline (T-1)", color="gray", linewidth=1, linestyle=":", alpha=0.5)
        
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # 底部: 置信度
        ax2.plot(plot_dates, confidences, label="Model Confidence Score", color="green", linewidth=1.5)
        ax2.fill_between(plot_dates, 0, confidences, color="green", alpha=0.1)
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # [修正] 添加时间戳和更多元信息到图表
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'Generated: {timestamp} | MAE: ${np.mean(np.abs(np.array(preds) - np.array(actuals))):.2f} | Corr: {corr:.3f}', 
                ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        plt.savefig("oil_price_prediction_full.png", dpi=300)
        print(f"\n✅ 增强版预测图已保存至 oil_price_prediction_full.png (生成时间: {timestamp})")

    explain_model_shap()          

    # Real-time inference
    predict_tomorrow(api_key=DEEPSEEK_API_KEY)  

