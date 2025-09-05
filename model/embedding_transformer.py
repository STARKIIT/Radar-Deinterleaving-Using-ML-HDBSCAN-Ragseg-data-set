import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding for temporal awareness."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """Enhanced transformer block with pre-normalization."""
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),  # Better activation than ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-normalization architecture
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + ff_out
        
        return x

class SelfAttentionPooling(nn.Module):
    """Self-attention based global pooling."""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x):  # (B, T, d_model)
        weights = torch.softmax(self.attention(x), dim=1)  # (B, T, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, d_model)
        return pooled

class PDWEncoder(nn.Module):
    """Transformer model for encoding PDW data into embeddings.
    
    This model processes radar pulse descriptor word (PDW) data and produces embeddings
    that can be used for clustering and classification tasks.
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, num_heads=4, num_layers=2, output_dim=32):
        """Initialize the PDW Encoder.
        
        Args:
            input_dim (int): Number of input features in PDW data
            hidden_dim (int): Size of transformer hidden layers
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            output_dim (int): Size of output embedding
        """
        super(PDWEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
                where batch_size is typically 1 for clustering
        
        Returns:
            torch.Tensor: Output embeddings of shape [batch_size, seq_len, output_dim]
        """
        # Project input features to transformer dimensions
        x = self.input_projection(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to output embedding dimension
        x = self.output_projection(x)
        
        return x
