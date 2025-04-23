import torch.nn as nn
import torch
from model.common import BasePytorchModel
class LSTMWrapper(nn.Module):
    def __init__(self, lstm_layer):
        super(LSTMWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = lstm_layer.to(self.device)

    def forward(self, x):
        # LSTM trả về (output, (hidden, cell))
        x = x.to(self.device)
        out, _ = self.lstm(x)
        return out

class LSTMModel(BasePytorchModel):
    def __init__(self, input_dim: int, hidden_dim: int | list[int], fc_dim: int, output_dim: int):
        super(LSTMModel, self).__init__()
        self.name = 'LSTMModel'
        # Nếu hidden_dim chỉ là int, chuyển thành list
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        
        # Tạo danh sách các LSTM layer và bọc chúng bằng LSTMWrapper
        lstm_layers = []
        input_dims = [input_dim] + hidden_dim
        for i in range(len(input_dims) - 1):
            lstm_layer = nn.LSTM(input_dims[i], input_dims[i+1], num_layers=1, batch_first=True)
            lstm_layers.append(LSTMWrapper(lstm_layer))
            
        self.lstm_stack = nn.Sequential(*lstm_layers).to(self.device)
        
        # Các lớp fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dims[-1], fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim)
        ).to(self.device)
        
    def forward(self, x):
        # x có shape: (batch_size, time_window, input_dim)
        x = x.to(self.device)
        out = self.lstm_stack(x)    # out có shape: (batch_size, time_window, hidden_dim[-1])
        out = out[:, -1, :]         # Lấy hidden state của bước cuối
        out = self.fc_layers(out)
        return out

