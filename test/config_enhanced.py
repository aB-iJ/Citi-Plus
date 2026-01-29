import torch

# 增强配置参数 - 更复杂的模型
class ConfigEnhanced:
    # 数据相关
    START_DATE = "2000-01-01"
    END_DATE = "2024-12-31"
    TICKER_OIL = "CL=F"
    REMOVE_EXTREME_OUTLIERS = True
    
    TICKERS_FACTORS = ["^GSPC", "DX-Y.NYB", "^VIX", "GC=F"]
    
    # 特征工程
    SEQ_LENGTH = 60    # 增加序列长度以捕捉更长期的模式
    PREDICT_STEPS = 1
    
    # 模型参数 - 大幅增强
    MODEL_TYPE = "EnhancedAttentionBiGRU"
    INPUT_DIM = 12
    HIDDEN_DIM = 256   # 从 128 增加到 256
    NUM_LAYERS = 4     # 从 2 增加到 4
    NUM_HEADS = 8      # 多头注意力机制
    OUTPUT_DIM = 3
    DROPOUT = 0.3      # 略微降低 dropout 让模型更能学习
    
    # 训练参数
    BATCH_SIZE = 64    # 增加批次大小
    EPOCHS = 150       # 增加训练轮次
    LEARNING_RATE = 0.0005  # 提高学习率
    PATIENCE = 25      # 增加耐心值
    
    # 路径
    MODEL_PATH = "best_oil_price_model_enhanced.pth"
    DATA_CACHE_PATH = "data/oil_data_merged.csv"

config_enhanced = ConfigEnhanced()
