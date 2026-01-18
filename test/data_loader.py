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
    计算 '新闻影响得分' 代理指标 (-1 到 1)。
    逻辑:
    1. 使用 VIX (波动率指数) 作为 '恐慌/不确定性' 指标。
    2. 使用日收益率确定方向。
    3. 归一化结果至 [-1, 1]。
    
    在生产系统中，这部分应该接入新闻标题的 NLP 情感评分。
    """
    # 假设 ^VIX_Close 存在 (Fear Index)
    if "^VIX_Close" in df.columns:
        vix = df["^VIX_Close"]
        # 将 VIX 归一化到 0-1 (相对于近期历史)
        vix_norm = (vix - vix.rolling(60).min()) / (vix.rolling(60).max() - vix.rolling(60).min() + 1e-6)
        
        # 市场方向 (使用油价自身变动作为方向)
        daily_ret = df["Oil_Close"].pct_change()
        direction = np.sign(daily_ret)
        
        # 新闻影响 = 带符号的 VIX 强度
        # 如果 VIX 高且价格跌 -> 负面新闻影响 (战争, 衰退恐慌)
        # 如果 VIX 高且价格涨 -> 正面新闻影响 (供应冲击新闻)
        news_score = direction * vix_norm
        
        # 填充 NaN
        news_score = news_score.fillna(0)
        
        # 限制在 -1, 1 之间
        news_score = news_score.clip(-1, 1)
        
        return news_score
    else:
        return pd.Series(0, index=df.index)

def preprocess_data(df):
    print("Preprocessing data...")
    df = df.copy()
    df = df.sort_index()
    df = df.ffill().bfill()
    
    # 1. 技术指标
    close = df['Oil_Close']
    
    # 简单移动平均 (SMA)
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    
    # RSI (相对强弱指数)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (指数平滑异同移动平均线)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # 布林带 (Bollinger Bands)
    roll_mean = close.rolling(window=20).mean()
    roll_std = close.rolling(window=20).std()
    df['Bollinger_Upper'] = roll_mean + (2 * roll_std)
    df['Bollinger_Lower'] = roll_mean - (2 * roll_std)
    
    # 2. 新闻影响因子 (-1 到 1)
    df['News_Impact'] = calculate_news_impact_score(df)
    
    # 3. 创建训练目标标签 (次日收盘价)
    df['Target_Price'] = df['Oil_Close'].shift(-config.PREDICT_STEPS)
    
    # 4. 波动率目标 (次日高低差)
    next_high = df['Oil_High'].shift(-config.PREDICT_STEPS)
    next_low = df['Oil_Low'].shift(-config.PREDICT_STEPS)
    df['Target_Volatility'] = next_high - next_low
    
    # 删除因移动/滞后或指标计算产生的 NaN
    df = df.dropna()
    
    return df

def get_processed_data():
    os.makedirs("data", exist_ok=True)
    load_from_cache = False
    
    if os.path.exists(config.DATA_CACHE_PATH):
        print(f"Checking cache: {config.DATA_CACHE_PATH}")
        try:
            df = pd.read_csv(config.DATA_CACHE_PATH, index_col=0, parse_dates=True)
            if len(df) > 50: # 简单的有效性检查
                print("Cache is valid.")
                load_from_cache = True
            else:
                print("Cache file is too small or empty. Ignoring.")
        except:
            print("Cache file is corrupt. Ignoring.")
            
    if not load_from_cache:
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
