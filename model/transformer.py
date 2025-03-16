import math
import torch
import torch.nn as nn
from common import BasePytorchModel

class PositionalEncoding(nn.Module):
    div_constant = 1e4
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Ma trận Positional Encoding
        pe = torch.zeros(max_len, d_model)
        # Vector vị trí 0 -> max_len -1
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Vector chuẩn hóa
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(PositionalEncoding.div_constant) / d_model))
        # Tính toán vị trí chẵn lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Thêm batch size
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Thêm pe vào tham số nhưng ko phải tham số huấn luyện
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        out = x + self.pe[:, :x.size(1)]
        return self.dropout(out)
    
class LocalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window: int):
        super(LocalAttention, self).__init__()
        self.num_heads = num_heads
        self.window = window
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape  # batch_first=True

        # Tạo mask riêng cho từng batch
        mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), device=x.device)  

        for i in range(seq_len):
            left = max(0, i - self.window)
            mask[:, i, left:i+1] = 0  # Chỉ cho phép nhìn thấy ngày hiện tại và quá khứ (t - window → t)

        # Điều chỉnh mask để phù hợp với MultiheadAttention
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
        mask = mask.reshape(batch_size * self.num_heads, seq_len, seq_len)  # (batch_size * num_heads, seq_len, seq_len)

        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_out

    
class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        super(StandardTransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        # Norm & Res & Activate
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, mask=None, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        x2 = self.self_attention(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window, dim_ff=2048, dropout=0.1):
        super(LocalTransformerEncoderLayer, self).__init__()
        self.self_attention = LocalAttention(embed_dim=d_model, num_heads=nhead, window=window)
        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        # Norm & Res & Activate
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, mask=None, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        x2 = self.self_attention(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=4):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.self_attention.embed_dim)
    
    def forward(self, x, mask=None, key_padding_mask=None):
        output = x
        for layer in self.layers:
            output = layer(output, mask=mask, key_padding_mask=key_padding_mask)
        return self.norm(output)
    
class StandardTransformerModel(BasePytorchModel):
    def __init__(self, optimizer_class: type[torch.optim.Optimizer], optimizer_params, 
                 input_dim, d_model, nhead, num_encoder_layers, 
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super(StandardTransformerModel, self).__init__(optimizer_class, optimizer_params)
        self.ndays = ndays
        
        # Embed
        self.input_linear = nn.Linear(input_dim, d_model)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        # Encoder Layer
        encoder_layer = StandardTransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Fully Connect
        self.fc = nn.Linear(d_model, output_dim * ndays)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = self.input_linear(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        out = self.encoder(x, mask=mask, key_padding_mask=key_padding_mask)
        out = out[:, -1, :]  # Lấy giá trị cuối cùng theo trục thời gian
        out = self.fc(out)  # (batch_size, output_dim * ndays)
        if out.shape[1] == self.ndays:
            return out
        return out.view(out.shape[0], self.ndays, -1)  # (batch, ndays, output_dim)
    
class LocalTransformerModel(StandardTransformerModel):
    def __init__(self, optimizer_class, optimizer_params, input_dim, 
                 d_model, nhead, num_encoder_layers, window,
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super().__init__(optimizer_class, optimizer_params, input_dim, d_model, nhead, 
                         num_encoder_layers, dim_ff, dropout, max_len, output_dim, ndays)
        self.window = window
        local_encoder_layer = LocalTransformerEncoderLayer(d_model, nhead, window, dim_ff, dropout)
        self.encoder = TransformerEncoder(local_encoder_layer, num_layers=num_encoder_layers)
