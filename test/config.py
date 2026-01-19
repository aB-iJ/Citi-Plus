import torch

# 配置参数
class Config:
    # 数据相关
    START_DATE = "2000-01-01"
    END_DATE = "2024-12-31" # 或者是当前日期
    TICKER_OIL = "CL=F" # WTI 原油 (目标)
    
    # 极端值处理模式
    # True: 将负油价视为缺失值 (NaN) -> 后续会被前值填充 (ffill) 覆盖
    #   优点: 模拟油价没有崩盘，按正常逻辑训练，模型更稳健
    # False: 保留真实的负油价
    #   缺点: 负值虽然真实，但在统计上是极端的异常值 (Outlier)，会破坏数据分布 (StandardScaler) 和梯度
    REMOVE_EXTREME_OUTLIERS = True
    
    # 辅助因子 Tickers
    # ^GSPC: 标普500 (宏观经济指标)
    # DX-Y.NYB: 美元指数 (通常与油价呈负相关)
    # ^VIX: CBOE 波动率指数 (市场恐慌/新闻情绪代理)
    # GC=F: 黄金 (避险资产)
    TICKERS_FACTORS = ["^GSPC", "DX-Y.NYB", "^VIX", "GC=F"]
    
    # 特征工程
    SEQ_LENGTH = 30    # [优化] 缩短序列，让模型更关注近期数据
    PREDICT_STEPS = 1  # 预测步长：预测未来 1 天
    
    # 模型参数
    MODEL_TYPE = "Transformer" 
    INPUT_DIM = 12     # 会自动覆盖
    HIDDEN_DIM = 128   # 隐藏层维度
    NUM_LAYERS = 3     # Transformer 层数
    NHEAD = 8          # 注意力头数
    DROPOUT = 0.15     # [优化] 进一步降低 Dropout，提高敏感度
    
    # 训练超参数
    BATCH_SIZE = 64    # [优化] 更大批次
    LR = 0.0005        # [优化] 提高学习率 (5e-4)
    EPOCHS = 200
    PATIENCE = 25      # 提前停止容忍度
    WEIGHT_DECAY = 1e-5 # [优化] 极低正则化
    
    # 爬虫/新闻配置
    USE_EXISTING_NEWS = True # True: 优先读取 crawled_news.json; False: 强制重新联网抓取
    SEARCH_ENGINE = "DuckDuckGo" # or "GoogleNews" (需安装库)
    
    # 模型保存路径
    MODEL_PATH = "best_oil_price_model.pth"
    DATA_CACHE_PATH = "data/oil_data_merged.csv"

config = Config()
