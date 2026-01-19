import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TemporalAttention(nn.Module):
    """
    时间注意力机制：让模型更关注近期数据
    使用可学习的时间衰减权重
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
        # 可学习的时间偏置：让近期数据权重更高
        self.time_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x: (Batch, Seq, d_model)
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x[:, -1:, :])  # 只用最后一个时间步做 Query
        K = self.key(x)
        V = self.value(x)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 添加时间偏置：越近的数据权重越高
        time_weights = torch.arange(seq_len, device=x.device).float()
        time_weights = (time_weights - seq_len + 1) * self.time_bias.abs()  # 近期为正
        scores = scores + time_weights.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        return output.squeeze(1), attn.squeeze(1)


class MultiScaleConv(nn.Module):
    """
    多尺度卷积：捕捉不同时间窗口的模式
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 不同大小的卷积核捕捉不同时间尺度的模式
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        # x: (Batch, Seq, Features) -> (Batch, Features, Seq) 用于一维卷积
        x = x.transpose(1, 2)
        
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        
        # 拼接不同尺度的特征
        out = torch.cat([c1, c3, c5, c7], dim=1)
        out = out.transpose(1, 2)  # 回到 (Batch, Seq, Features)
        
        return self.norm(out)


class OilPriceTransformer(nn.Module):
    """
    改进版油价预测模型 - 专门针对滞后问题优化
    
    核心改进：
    1. 多尺度卷积：捕捉不同时间窗口的价格模式
    2. 时间注意力：强制模型关注近期数据
    3. 残差学习：预测变化量而非绝对值
    4. 双向信息流：LSTM + Transformer 结合
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.2, nhead=8):
        super(OilPriceTransformer, self).__init__()
        
        self.d_model = hidden_dim
        self.input_dim = input_dim
        
        # 1. 多尺度特征提取
        self.multi_scale_conv = MultiScaleConv(input_dim, hidden_dim)
        
        # 2. Input Embedding (处理卷积输出)
        self.input_embed = nn.Linear(hidden_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # 3. 双向 LSTM 捕捉序列依赖
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 5. 时间注意力池化
        self.temporal_attn = TemporalAttention(self.d_model)
        
        # 6. 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
        )
        
        # 7. Output Heads - 预测变化量
        # Return Head (核心：预测收益率)
        self.fc_return = nn.Sequential(
            nn.Linear(self.d_model // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 不确定性 Head
        self.fc_sigma = nn.Sequential(
            nn.Linear(self.d_model // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 波动率 Head
        self.fc_vol = nn.Sequential(
            nn.Linear(self.d_model // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # 8. 直接从最近数据提取的快速通道 (Skip Connection)
        # 这个通道让模型可以直接从最近几天的数据做快速反应
        self.fast_track = nn.Sequential(
            nn.Linear(input_dim * 5, 64),  # 最近5天的原始特征
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 融合权重
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 主路径权重
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        batch_size, seq_len, _ = x.shape
        
        # === 快速通道：直接从最近5天数据提取信号 ===
        recent_days = min(5, seq_len)
        recent_data = x[:, -recent_days:, :].reshape(batch_size, -1)
        # 如果长度不足，进行填充
        if recent_days < 5:
            pad = torch.zeros(batch_size, (5 - recent_days) * self.input_dim, device=x.device)
            recent_data = torch.cat([pad, recent_data], dim=1)
        fast_pred = self.fast_track(recent_data)
        
        # === 主路径：深度特征提取 ===
        # 1. 多尺度卷积
        x_conv = self.multi_scale_conv(x)
        
        # 2. 嵌入 + 位置编码
        x_embed = self.input_embed(x_conv)
        x_embed = self.input_norm(x_embed)
        x_embed = x_embed.transpose(0, 1)
        x_embed = self.pos_encoder(x_embed)
        x_embed = x_embed.transpose(0, 1)
        
        # 3. LSTM 处理
        lstm_out, _ = self.lstm(x_embed)
        
        # 4. Transformer 处理
        trans_out = self.transformer_encoder(lstm_out)
        
        # 5. 时间注意力池化
        attn_out, attn_weights = self.temporal_attn(trans_out)
        
        # 6. 结合 LSTM 最后状态和注意力输出
        lstm_last = lstm_out[:, -1, :]
        combined = torch.cat([attn_out, lstm_last], dim=-1)
        
        # 7. 特征融合
        fused = self.fusion(combined)
        
        # 8. 预测
        main_pred = self.fc_return(fused)
        
        # === 融合快速通道和主路径 ===
        alpha = torch.sigmoid(self.alpha)
        pred_return = alpha * main_pred + (1 - alpha) * fast_pred
        
        log_var = self.fc_sigma(fused)
        vol = self.fc_vol(fused)
        
        return pred_return, log_var, vol, attn_weights
