import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBiGRU(nn.Module):
    """
    纯分类模型：只预测涨/跌方向，不预测具体价格
    这样可以完全消除"滞后性"问题
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(AttentionBiGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Layer
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # [核心] Direction Classification Head - 主要输出
        # 输出上涨概率 (0-1)
        self.fc_direction = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Confidence Head - 预测置信度
        self.fc_confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        gru_out, _ = self.gru(x)
        
        # Attention
        attn_scores = self.attention_net(gru_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(gru_out * attn_weights, dim=1)
        
        # 输出
        direction = self.fc_direction(context_vector)  # 上涨概率
        confidence = self.fc_confidence(context_vector)  # 置信度
        
        return direction, confidence, attn_weights
