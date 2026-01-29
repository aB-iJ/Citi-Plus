import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_linear(context)
        return output, attn_weights.mean(dim=1)  # 返回平均注意力权重


class EnhancedAttentionBiGRU(nn.Module):
    """
    增强版模型:
    - 更深的 BiGRU (4层)
    - 多头注意力机制
    - 残差连接
    - LayerNorm
    - 更大的隐藏维度
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # BiGRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, num_layers=1, 
                   batch_first=True, bidirectional=True, dropout=0)
            for _ in range(num_layers)
        ])
        
        # Projection layers for residual (bidirectional doubles the size)
        self.residual_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # LayerNorm for each GRU layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Multi-head attention
        self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Additional attention layer (simple)
        self.attn_layer = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)  # [B, Seq, Hidden]
        
        # Stack GRU layers with residuals
        for i, (gru, proj, norm) in enumerate(zip(self.gru_layers, self.residual_projs, self.layer_norms)):
            residual = x
            x, _ = gru(x)  # [B, Seq, Hidden*2]
            x = proj(x)     # [B, Seq, Hidden]
            x = norm(x + residual)  # Residual + LayerNorm
            x = self.dropout(x)
        
        # Multi-head attention
        attn_out, attn_weights = self.multihead_attn(x)
        x = x + attn_out  # Residual connection
        
        # Simple attention for final representation
        attn_scores = self.attn_layer(x)  # [B, Seq, 1]
        attn_scores = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_scores * x, dim=1)  # [B, Hidden]
        
        # Predictions
        direction = self.direction_head(context)
        confidence = self.confidence_head(context)
        
        return direction, confidence, attn_weights

# 保持原来的简单模型用于比较
class AttentionBiGRU(nn.Module):
    """简单基线模型"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                         batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attn_layer = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        
        attn_scores = self.attn_layer(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        direction = self.direction_head(context)
        confidence = self.confidence_head(context)
        
        return direction, confidence, attn_weights
