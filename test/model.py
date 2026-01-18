import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(AttentionBiGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional GRU
        # Bi-Directional 意味着 hidden_size 输出会翻倍
        # Input: (Batch, Seq_Len, Input_Dim)
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Layer
        # 将 GRU 的输出 (Hidden*2) 映射到一个权重分数
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output Heads (Regression)
        # 输入维度是 Context Vector (Hidden*2)
        
        # 1. Price Head
        self.fc_price = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LeakyReLU(), # LeakyReLU 在金融回归中通常比 ReLU 表现更好（保留负区间微小梯度）
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # 2. Uncertainty Head (Aleatoric Uncertainty)
        self.fc_sigma = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Volatility Head (Range Prediction)
        self.fc_vol = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # 保证波动率为正
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        
        # 1. GRU Forward
        # gru_out: (Batch, Seq, Hidden * 2)
        # h_n: (Num_Layers * 2, Batch, Hidden)
        gru_out, _ = self.gru(x)
        
        # 2. Compute Attention Weights
        # 这里的目的是给序列中每一步赋予权重
        # scores: (Batch, Seq, 1)
        attn_scores = self.attention_net(gru_out)
        attn_weights = F.softmax(attn_scores, dim=1) #(Batch, Seq, 1)
        
        # 3. Compute Context Vector
        # 对序列进行加权求和，融合全局信息
        # context: (Batch, Hidden * 2)
        context_vector = torch.sum(gru_out * attn_weights, dim=1)
        
        # 4. Predict
        price = self.fc_price(context_vector)
        log_var = self.fc_sigma(context_vector)
        vol = self.fc_vol(context_vector)
        
        return price, log_var, vol, attn_weights
