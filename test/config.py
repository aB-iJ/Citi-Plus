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
    SEQ_LENGTH = 60    # 序列长度：回看过去 60 天的数据进行预测
    PREDICT_STEPS = 1  # 预测步长：预测未来 1 天
    
    # 模型参数
    MODEL_TYPE = "AttentionBiGRU" 
    INPUT_DIM = 12     # 会自动覆盖
    HIDDEN_DIM = 128   # 隐藏层维度
    NUM_LAYERS = 2     # GRU 层数
    OUTPUT_DIM = 3
    DROPOUT = 0.4      # 随机失活率（防止过拟合）
    
    # 训练参数
    BATCH_SIZE = 32    # 批次大小
    EPOCHS = 100
    LEARNING_RATE = 0.0001 # 初始学习率
    PATIENCE = 15      # 早停机制 (Early Stopping)
    
    # 路径
    MODEL_PATH = "best_oil_price_model.pth"
    DATA_CACHE_PATH = "data/oil_data_merged.csv"

config = Config()
