import yfinance as yf
import pandas as pd
import numpy as np
import ta
import akshare as ak
from config import config
import os

def parse_and_clean_df(csv_path):
    """
    读取 CSV 文件，智能解析日期（支持 'Jan 16 2026' 或 '2026-01-16'），
    清洗数值列（处理 '-' 等异常字符），并按日期排序。
    """
    try:
        # 首先读取文件，暂不解析日期，以便手动处理特殊格式
        df = pd.read_csv(csv_path)
        
        # 1. 解析日期
        # 尝试默认推断，如果失败或包含字符串，则强制自定义解析
        if "Date" in df.columns:
            # 检查样本以猜测格式，但 to_datetime 通常足够智能
            # 例如 "Jan 16 2026" -> %b %d %Y
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date"])
            df = df.set_index("Date").sort_index()
        else:
            print(f"Error: No 'Date' column in {csv_path}")
            return None

        # 2. 清洗非数值数据 (如 '-')
        # 遍历列以清洗垃圾字符
        for col in df.columns:
            # 如果列类型是对象，很可能包含 '-' 或字符串
            if df[col].dtype == 'object':
                # 将 '-' 替换为 NaN
                df[col] = df[col].replace('-', np.nan).replace('null', np.nan)
                # 移除数字中的逗号 (例如 "1,234.56")
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                # 转换为数值，失败则变成 NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error reading/cleaning {csv_path}: {e}")
        return None

def load_from_local_files():
    """
    尝试从 'input_data/' 文件夹加载原始 CSV 文件。
    预期文件名: CL=F.csv, ^GSPC.csv 等。
    """
    input_dir = "input_data"
    if not os.path.exists(input_dir):
        return None
        
    print(f"Checking for local CSV files in {input_dir}...")
    tickers = [config.TICKER_OIL] + config.TICKERS_FACTORS
    
    df_main = pd.DataFrame()
    
    # 1. 加载原油数据 (目标)
    oil_path = None
    potential_names = [config.TICKER_OIL, config.TICKER_OIL.replace("=", "_").replace("^", "")]
    # 同时也检查常用的安全文件名
    potential_names += ["CL=F", "CL_F", "Oil"] 
    
    oil_df = None
    for name in potential_names:
        p = os.path.join(input_dir, f"{name}.csv")
        if os.path.exists(p):
            print(f"Found local file: {p}")
            oil_df = parse_and_clean_df(p)
            break
            
    if oil_df is None:
        print(f"Missing local file for target {config.TICKER_OIL}")
        return None
        
    # 标准化原油列名
    # Yahoo CSV 列: Date, Open, High, Low, Close, Adj Close, Volume
    cols_map = {"Close": "Oil_Close", "Open": "Oil_Open", "High": "Oil_High", "Low": "Oil_Low", "Volume": "Oil_Volume"}
    # 灵活的列重命名 (不区分大小写)
    oil_df.columns = [c.title() for c in oil_df.columns] # 确保首字母大写 (Close, Open...)
    
    oil_df = oil_df.rename(columns=cols_map)
    # 仅保留需要的列
    available_cols = [c for c in cols_map.values() if c in oil_df.columns]
    df_main = oil_df[available_cols]

    # FIX: 处理负油价 (2020年4月黑天鹅事件)
    # 模式选择: 根据 config.py 中的 REMOVE_EXTREME_OUTLIERS 决定
    if "Oil_Close" in df_main.columns:
        neg_mask = df_main["Oil_Close"] < 0
        neg_count = neg_mask.sum()
        
        if neg_count > 0:
            if config.REMOVE_EXTREME_OUTLIERS:
                print(f"Warning: Found {neg_count} negative oil prices. Mode=REMOVE. Setting to NaN (will be ffilled).")
                # 将所有相关列的负值（行）设为 NaN
                # 这样后续的 ffill() 会用 crash 前的最后正常价格覆盖这段时间
                cols_to_clean = [c for c in ["Oil_Close", "Oil_Low", "Oil_High", "Oil_Open"] if c in df_main.columns]
                for c in cols_to_clean:
                   df_main.loc[df_main[c] < 0, c] = np.nan
                   
            else:
                print(f"Warning: Found {neg_count} negative oil prices. Mode=KEEP. Treating as valid extreme data.")
                # 不做任何处理，保留原样

    # 2. 加载辅助因子
    for t in config.TICKERS_FACTORS:
        factor_df = None
        potential_names = [t, t.replace("=", "_").replace("^", "")]
        for name in potential_names:
            p = os.path.join(input_dir, f"{name}.csv")
            if os.path.exists(p):
                print(f"Found local file for factor {t}: {p}")
                factor_df = parse_and_clean_df(p)
                break
        
        col_name = f"{t}_Close"
        if factor_df is not None:
             # 灵活的列检查
            factor_df.columns = [c.title() for c in factor_df.columns]
            
            if "Close" in factor_df.columns:
                # 合并逻辑: 基于原油索引进行左连接 (Left Join)
                # 这会自动对齐日期，并在日期不匹配的地方填入 NaN
                df_main = df_main.join(factor_df["Close"].rename(col_name), how="left")
            else:
                df_main[col_name] = np.nan
        else:
            print(f"Warning: Local file for {t} not found. Filling NaN.")
            df_main[col_name] = np.nan
            
    # 3. 处理缺失数据（不使用模拟数据）
    # 策略: 前向填充 (FFill - 用最近一次已知数据填充后续缺失)
    # 这是金融领域的标准做法："在有新交易之前，价格保持不变"
    df_main = df_main.ffill()
    
    # 对数据集开头可能存在的 NaN 进行后向填充 (Backfill)
    df_main = df_main.bfill()
    
    # 如果仍然存在 NaN (例如整列都是空的)，删除或填充0?
    # 如果数据很关键最好删除不完整的行，但这里我们尽量宽容。
    if df_main.isnull().values.any():
        print("Warning: Still found NaNs after fill. Dropping incomplete rows.")
        df_main = df_main.dropna()

    print(f"Local data loaded & cleaned. Shape: {df_main.shape}")
    return df_main

def fetch_yfinance_data():
    # 优先级 1: 尝试本地 CSV
    local_df = load_from_local_files()
    if local_df is not None:
        return local_df

    print("Fetching data from yfinance...")
    tickers = [config.TICKER_OIL] + config.TICKERS_FACTORS
    
    try:
        # 一次性获取所有数据
        data = yf.download(tickers, start=config.START_DATE, end=config.END_DATE, group_by='ticker', progress=True)
        
        # 检查数据是否为空或缺少关键列
        if data is None or data.empty:
            print("ERROR: yfinance returned empty data")
            exit(1)

        df_main = pd.DataFrame()
        
        # 安全获取数据的辅助函数
        def get_ticker_data(ticker_name):
            if isinstance(data.columns, pd.MultiIndex):
                if ticker_name in data.columns.levels[0]:
                    return data[ticker_name]
            # 如果结构不同或是单独下载，这里是备用逻辑
            return None

        # 处理原油数据 (目标)
        oil_df = get_ticker_data(config.TICKER_OIL)
        if oil_df is None or oil_df.empty:
             print(f"ERROR: Target ticker {config.TICKER_OIL} data missing")
             exit(1)

        oil_df = oil_df.copy()
        # 重命名列为标准格式
        # 注意: yfinance 列通常为 [Open, High, Low, Close, Volume]
        cols_map = {"Close": "Oil_Close", "Open": "Oil_Open", "High": "Oil_High", "Low": "Oil_Low", "Volume": "Oil_Volume"}
        
        oil_df = oil_df.rename(columns=cols_map)
        df_main = oil_df[list(cols_map.values())] # 仅保留重命名后的列
        
        # 处理辅助因子
        for t in config.TICKERS_FACTORS:
            f_data = get_ticker_data(t)
            col_name = f"{t}_Close"
            if f_data is not None and not f_data.empty:
                df_main[col_name] = f_data["Close"]
            else:
                # 如果特定因子失败，尝试单独获取或填充
                try:
                    tmp = yf.download(t, start=config.START_DATE, end=config.END_DATE, progress=False)
                    if not tmp.empty:
                        df_main[col_name] = tmp["Close"]
                    else:
                        df_main[col_name] = np.nan # 稍后填充
                except:
                    df_main[col_name] = np.nan

        # 检查是否有足够的行数
        if len(df_main) < 10:
             print("ERROR: Insufficient data rows retrieved")
             exit(1)
             
        # 处理缺失的因子数据
        df_main = df_main.ffill().bfill()
        return df_main

    except Exception as e:
        print(f"Error fetching YFinance Data: {e}")
        exit(1)

def fetch_akshare_data():
    """
    集成 AkShare 的示例。
    尝试获取宏观经济指标。
    """
    print("正在从 AkShare 获取数据 (宏观/新闻代理)...")
    try:
        # 该函数作为本地/中国特定数据的集成点。
        return pd.DataFrame()
    except Exception as e:
        print(f"AkShare fetch failed/skipped: {e}")
        return pd.DataFrame()

def calculate_news_impact_score(df):
    """
    [重构] 计算"新闻影响"代理指标 - 不使用未来信息！
    
    之前的问题：使用 shift(-1) 获取"明天的涨跌"，这是数据泄露 (Data Leakage)！
    模型在训练时学会了依赖这个"作弊"特征，一旦推理时没有未来信息，就严重滞后。
    
    新策略：使用纯历史信息构建情绪代理
    1. VIX 变化率 (恐慌指数的变化)
    2. 近期动量 (价格走势)
    3. 成交量异常 (可能暗示消息面变化)
    """
    df = df.copy()
    
    # 1. VIX 变化率 (恐慌情绪的变化速度)
    if "^VIX_Close" in df.columns:
        vix = df["^VIX_Close"]
        # VIX 的日变化率
        vix_change = vix.pct_change().fillna(0)
        # 归一化到 -1 ~ 1
        vix_score = vix_change.clip(-0.2, 0.2) * 5  # 20% 变化 -> ±1
    else:
        vix_score = pd.Series(0, index=df.index)
    
    # 2. 短期动量 (过去3天的累计收益率)
    if "Oil_Close" in df.columns:
        returns_3d = df["Oil_Close"].pct_change(3).fillna(0)
        momentum_score = returns_3d.clip(-0.1, 0.1) * 10  # 10% 变化 -> ±1
    else:
        momentum_score = pd.Series(0, index=df.index)
    
    # 3. 成交量异常 (相对于20日均量的偏离)
    if "Oil_Volume" in df.columns:
        vol = df["Oil_Volume"]
        vol_ma = vol.rolling(window=20, min_periods=1).mean()
        vol_ratio = (vol / (vol_ma + 1e-10)) - 1  # 超过均值的比例
        vol_score = vol_ratio.clip(-2, 2) * 0.25  # 2倍量 -> 0.5
    else:
        vol_score = pd.Series(0, index=df.index)
    
    # 4. 组合得分 (加权平均)
    # VIX 变化最能反映市场情绪，给更高权重
    news_score = 0.5 * vix_score + 0.3 * momentum_score + 0.2 * vol_score
    
    # 最终裁剪到 -1 ~ 1
    news_score = news_score.clip(-1, 1).fillna(0)
    
    return news_score

def preprocess_data(df):
    print("Preprocessing data...")
    df = df.copy()
    df = df.sort_index()
    df = df.ffill().bfill()
    
    # [核心修改: 消除滞后] 
    # 强制让模型预测 "对数收益率" (Log Returns): ln(P_t / P_{t-1})
    
    # 1. 计算对数收益率
    df['Log_Return'] = np.log(df['Oil_Close'] / df['Oil_Close'].shift(1))
    df['Log_Return'] = df['Log_Return'].fillna(0) # 第一天为0

    # 2. 技术指标
    close = df['Oil_Close']
    
    # Simple Moving Average
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    
    # [优化] 添加更多动量/变化敏感指标
    
    # [新增] 极速动量 (Velocity & Acceleration)
    # 滞后性修复核心：引入二阶差分 (加速度)，这通常是价格变化的领先指标
    df['Velocity'] = df['Log_Return'] # 一阶 (速度)
    df['Acceleration'] = df['Log_Return'] - df['Log_Return'].shift(1) # 二阶 (加速度)
    
    # 价格变化率 (Rate of Change) - 5日 -> 改为更短的 3日，提高敏感度
    df['ROC_3'] = (close - close.shift(3)) / (close.shift(3) + 1e-10) * 100
    
    # 动量 RSI - 缩短周期 10 -> 6
    df['Momentum_6_RSI'] = ta.momentum.RSIIndicator(close, window=6).rsi() 
    
    # 短期波动率 (5日标准差) -> 标准化: Vol / Close
    df['Volatility_5'] = close.rolling(window=5).std() / (close + 1e-10)
    
    # 价格均线偏离度 (保留短期5日，移除相对滞后的20日作为主特征)
    df['Price_SMA5_Ratio'] = close / (df['SMA_5'] + 1e-10) - 1
    # df['Price_SMA20_Ratio'] = close / (df['SMA_20'] + 1e-10) - 1 # 禁用长周期
    
    # RSI - 缩短周期 14 -> 6 (快速 RSI)
    # Fast RSI reacts much faster to price reversals
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_6'] = 100 - (100 / (1 + rs))
    
    # MACD -> 标准化 MACD / Close
    # 使用更快的参数 (6, 13, 5) 替代标准的 (12, 26, 9)
    exp1 = close.ewm(span=6, adjust=False).mean()
    exp2 = close.ewm(span=13, adjust=False).mean()
    macd_raw = exp1 - exp2
    df['MACD_Norm'] = macd_raw / (close + 1e-10)
    
    macd_signal = macd_raw.ewm(span=5, adjust=False).mean()
    df['MACD_Signal_Norm'] = macd_signal / (close + 1e-10)
    
    df['MACD_Hist_Norm'] = df['MACD_Norm'] - df['MACD_Signal_Norm']

    
    # Bollinger Bands
    roll_mean = close.rolling(window=20).mean()
    roll_std = close.rolling(window=20).std()
    df['Bollinger_Upper'] = roll_mean + (2 * roll_std)
    df['Bollinger_Lower'] = roll_mean - (2 * roll_std)
    # Bollinger %B (价格在布林带中的位置) - 这是一个很好的归一化指标
    df['Bollinger_PctB'] = (close - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'] + 1e-10)
    
    # Volume 归一化: Vol / SMA_Vol_20
    if 'Oil_Volume' in df.columns:
         vol_sma = df['Oil_Volume'].rolling(window=20).mean()
         df['Volume_Ratio'] = df['Oil_Volume'] / (vol_sma + 1e-10)
    else:
         df['Volume_Ratio'] = 0

    df = df.ffill().bfill()


    # 3. [重要] 定义 Target (预测目标)
    # Target(t) = Log_Return(t+PREDICT_STEPS)
    df['Target_Return'] = df['Log_Return'].shift(-config.PREDICT_STEPS)
    
    # 保留 Target_Price 用于评估
    df['Target_Price'] = df['Oil_Close'].shift(-config.PREDICT_STEPS)
    
    # 移除最后 N 行
    df = df.iloc[:-config.PREDICT_STEPS]
    
    # 4. 新闻影响因子 (-1 到 1)
    df['News_Impact'] = calculate_news_impact_score(df)
    
    # 5. 选择特征列
    # [优化] 仅保留平稳特征 (Stationary Features)，移除原始价格 (Raw Prices) 以消除滞后性
    feature_cols = [
        'Log_Return',           # 速度 (V)
        'Acceleration',         # 加速度 (A) - 关键领先指标
        'Volume_Ratio',         # 相对成交量
        'ROC_3',                # 快速变化率
        'Momentum_6_RSI',       # 快速 RSI 动量
        'Volatility_5',         # 相对波动率
        'Price_SMA5_Ratio',     # 短期乖离率
        'RSI_6',                # 快速 RSI
        'MACD_Norm',            # 快速 MACD
        'MACD_Hist_Norm',       # 快速 MACD 柱
        'Bollinger_PctB',       # 布林位置
        'News_Impact'           # 新闻
    ]
    
    # 添加外部因子 - 必须转换为收益率或相对值
    feature_cols_extra = []
    for t in config.TICKERS_FACTORS:
        col = f"{t}_Close"
        if col in df.columns:
            # 自动转换为 Log Return
            ret_col = f"{col}_LogRet"
            df[ret_col] = np.log(df[col] / df[col].shift(1))
            df[ret_col] = df[ret_col].fillna(0)
            feature_cols_extra.append(ret_col)
    
    feature_cols.extend(feature_cols_extra)

            
    # 重新整理
    # 计算 Target_Volatility (High - Low)
    # 如果 High/Low 不存在，用 ATR (rolling std) 代替
    if 'Oil_High' in df.columns and 'Oil_Low' in df.columns:
        # Volatility for the target day
        vol = df['Oil_High'] - df['Oil_Low']
        df['Target_Volatility'] = vol.shift(-config.PREDICT_STEPS)
    else:
        # Simple Proxy
        df['Target_Volatility'] = df['Log_Return'].rolling(5).std().shift(-config.PREDICT_STEPS)
    
    # [修复] 必须保留 Oil_Close 用于后续评估和过滤，即使不作为特征
    df_final = df[feature_cols + ['Target_Return', 'Target_Price', 'Target_Volatility', 'Oil_Close']].copy()
    
    if config.REMOVE_EXTREME_OUTLIERS:
         df_final = df_final[df_final['Oil_Close'] > 0]
         
    # 再次去除 NaN 的行
    df_final = df_final.dropna()
         
    return df_final

def get_processed_data():
    os.makedirs("data", exist_ok=True)
    # [修改] 强制每次重新生成数据，不再使用缓存读取，以确保特征工程的修改立即可见
    load_from_cache = False
    
    # 优先级 1: 本地文件
    df = load_from_local_files()
        
    # 优先级 2: API (如果没有本地文件)
    if df is None or len(df) == 0:
        print("No local files found. Attempting YFinance download...")
        df = fetch_yfinance_data()

    if df is not None and len(df) > 0:
        df = preprocess_data(df)
        print(f"Saving data to cache: {config.DATA_CACHE_PATH}")
        df.to_csv(config.DATA_CACHE_PATH)
    else:
        print("Error: Could not obtain data from Local Files or YFinance.")
        return None
        
    return df

if __name__ == "__main__":
    df = get_processed_data()
    print(df.head())
    print(df.describe())
