import math
import torch
import torch.nn as nn
from model.common import BasePytorchModel

class PositionalEncoding(nn.Module):
    div_constant = 1e4
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(p=dropout).to(self.device)
        
        # Ma trận Positional Encoding
        pe = torch.zeros(max_len, d_model, device=self.device)
        # Vector vị trí 0 -> max_len -1
        position = torch.arange(0, max_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        # Vector chuẩn hóa
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * (-math.log(PositionalEncoding.div_constant) / d_model))
        # Tính toán vị trí chẵn lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Thêm batch size
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Thêm pe vào tham số nhưng ko phải tham số huấn luyện
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        x = x.to(self.device)
        out = x + self.pe[:, :x.size(1)]
        return self.dropout(out)
    
class LocalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window: int):
        super(LocalAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.window = window
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        batch_size, seq_len, embed_dim = x.shape  # batch_first=True

        # Tạo mask riêng cho từng batch
        mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), device=x.device)  

        for i in range(seq_len):
            left = max(0, i - self.window)
            right = min(seq_len, i + self.window + 1)
            mask[:, i, left:right] = 0  

        # Điều chỉnh mask để phù hợp với MultiheadAttention
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
        mask = mask.reshape(batch_size * self.num_heads, seq_len, seq_len)  # (batch_size * num_heads, seq_len, seq_len)

        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_out
    
class CausalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(CausalAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        batch_size, seq_len, embed_dim = x.shape
        mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), device=self.device)

        for i in range(seq_len):
            mask[:, i, :i+1] = 0  
        
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        mask = mask.reshape(batch_size * self.num_heads, seq_len, seq_len)
        
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_out

class RestrictedCausalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window: int):
        super(RestrictedCausalAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.window = window
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, seq_len, embed_dim = x.shape  # batch_first=True

        # Tạo mask riêng cho từng batch
        mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), device=x.device)  

        for i in range(seq_len):
            left = max(0, i - self.window)
            right = min(seq_len, i + 1)
            mask[:, i, left:right] = 0  

        # Điều chỉnh mask để phù hợp với MultiheadAttention
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_len, seq_len)
        mask = mask.reshape(batch_size * self.num_heads, seq_len, seq_len)  # (batch_size * num_heads, seq_len, seq_len)

        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_out
    
class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        super(StandardTransformerEncoderLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True).to(self.device)
        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_ff).to(self.device)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model).to(self.device)
        self.norm1 = nn.LayerNorm(d_model).to(self.device)
        self.norm2 = nn.LayerNorm(d_model).to(self.device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
        # x: (batch, seq_len, d_model)
        x2 = self.self_attention(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class LocalTransformerEncoderLayer(StandardTransformerEncoderLayer):
    def __init__(self, d_model, nhead, window, dim_ff=2048, dropout=0.1):
        super(LocalTransformerEncoderLayer, self).__init__(d_model, nhead, dim_ff, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attention = LocalAttention(embed_dim=d_model, num_heads=nhead, window=window).to(self.device)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
        # x: (batch, seq_len, d_model)
        x2 = self.self_attention(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class CausalTransformerEncoderLayer(StandardTransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        super(CausalTransformerEncoderLayer, self).__init__(d_model, nhead, dim_ff, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attention = CausalAttention(embed_dim=d_model, num_heads=nhead).to(self.device)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
        # x: (batch, seq_len, d_model)
        x2 = self.self_attention(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
class RestrictedCausalTransformerEncoderLayer(StandardTransformerEncoderLayer):
    def __init__(self, d_model, nhead, window, dim_ff=2048, dropout=0.1):
        super(RestrictedCausalTransformerEncoderLayer, self).__init__(d_model, nhead, dim_ff, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attention = RestrictedCausalAttention(embed_dim=d_model, num_heads=nhead, window=window).to(self.device)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList([encoder_layer.to(self.device) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.self_attention.embed_dim).to(self.device)
    
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
        output = x
        for layer in self.layers:
            output = layer(output, mask=mask, key_padding_mask=key_padding_mask)
        return self.norm(output)
    
class StandardTransformerModel(BasePytorchModel):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super(StandardTransformerModel, self).__init__()
        self.ndays = ndays
        
        # Embed
        self.input_linear = nn.Linear(input_dim, d_model).to(self.device)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(self.device)
        # Encoder Layer
        encoder_layer = StandardTransformerEncoderLayer(d_model, nhead, dim_ff, dropout).to(self.device)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers).to(self.device)
        
        # Fully Connect
        self.fc = nn.Linear(d_model, output_dim * ndays).to(self.device)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x = x.to(self.device)
        x = self.input_linear(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        out = self.encoder(x, mask=mask, key_padding_mask=key_padding_mask)
        out = out[:, -1, :]  # Lấy giá trị cuối cùng theo trục thời gian
        out = self.fc(out)  # (batch_size, output_dim * ndays)
        if out.shape[1] == self.ndays:
            return out
        return out.view(out.shape[0], self.ndays, -1)  # (batch, ndays, output_dim)
    
class LocalTransformerModel(StandardTransformerModel):
    def __init__(self, input_dim, 
                 d_model, nhead, num_encoder_layers, window,
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super().__init__(input_dim, d_model, nhead, 
                         num_encoder_layers, dim_ff, dropout, max_len, output_dim, ndays)
        self.window = window
        local_encoder_layer = LocalTransformerEncoderLayer(d_model, nhead, window, dim_ff, dropout).to(self.device)
        self.encoder = TransformerEncoder(local_encoder_layer, num_layers=num_encoder_layers).to(self.device)

class CausalTransformerModel(StandardTransformerModel):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super().__init__(input_dim, d_model, nhead, 
                         num_encoder_layers, dim_ff, dropout, 
                         max_len, output_dim, ndays)
        causal_encoder_layer = CausalTransformerEncoderLayer(d_model, nhead, dim_ff, dropout).to(self.device)
        self.encoder = TransformerEncoder(causal_encoder_layer, num_layers=num_encoder_layers).to(self.device)
        
class RestrictedCausalTransformerModel(StandardTransformerModel):
    def __init__(self, input_dim, 
                 d_model, nhead, num_encoder_layers, window,
                 dim_ff=2048, dropout=0.1, max_len=100, output_dim=1, ndays=5):
        super().__init__(input_dim, d_model, nhead, 
                         num_encoder_layers, dim_ff, dropout, max_len, output_dim, ndays)
        self.window = window
        r_causal_encoder_layer = RestrictedCausalTransformerEncoderLayer(d_model, nhead, window, dim_ff, dropout).to(self.device)
        self.encoder = TransformerEncoder(r_causal_encoder_layer, num_layers=num_encoder_layers).to(self.device)
