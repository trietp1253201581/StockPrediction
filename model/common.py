import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class supported_metrics:
    def __init__(self, metric, name: str):
        self.metric = metric
        self.name = name
        
    def __str__(self):
        return self.name
    
    def __call__(self, y_true, y_pred):
        return self.metric(y_true, y_pred)
    
mse = supported_metrics(mean_squared_error, 'mse')
mae = supported_metrics(mean_absolute_error, 'mae')
corr = supported_metrics(np.corrcoef, 'corr')

# Lớp BaseModel trừu tượng
class BaseModel(ABC):
    @abstractmethod
    def train_model(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    def evaluate_model(self, y_true, y_pred, metric: supported_metrics):
        ndays = y_true.shape[1]
        perdays = np.array([metric(y_true[:, i], y_pred[:, i]) for i in range(ndays)])
        return perdays, perdays.mean()

# Lớp BasePytorchModel chứa các hàm hỗ trợ dùng chung cho các mô hình deep learning PyTorch
class BasePytorchModel(nn.Module, BaseModel):
    def __init__(self):
        super(BasePytorchModel, self).__init__()
    
    @staticmethod
    def make_loader(X, y, batch_size=32, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        # Nếu y có dạng (L,) thì unsqueeze để có shape (L, 1); nếu y là multi-step (L, ndays) thì giữ nguyên
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def train_model(self, X, y, loss_fn, num_epochs, lr, batch_size=32, X_val=None, y_val=None):
        """
        Huấn luyện mô hình với loss function được truyền vào.
        
        Parameters:
          - X, y: Dữ liệu huấn luyện (có thể là cho dự báo 1 ngày hay multi-step).
          - loss_fn: Hàm loss (ví dụ: nn.MSELoss()).
          - num_epochs: Số epoch huấn luyện.
          - lr: Learning rate.
          - batch_size: Kích thước batch.
          - X_val, y_val (tuỳ chọn): Dữ liệu validation để tính loss sau mỗi epoch.
        """
        train_loader = BasePytorchModel.make_loader(X, y, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(train_loader.dataset)
            
            # Tính loss trên tập validation nếu có
            if X_val is not None and y_val is not None:
                val_loss = self.valid_model(X_val, y_val, loss_fn, batch_size=batch_size)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
    
    def valid_model(self, X, y, loss_fn, batch_size=32):
        """
        Tính loss trên tập validation (hoặc test) với loss_fn được truyền vào.
        Đây là phương thức chuyên dùng cho việc đánh giá tập validation.
        """
        loader = BasePytorchModel.make_loader(X, y, batch_size=batch_size, shuffle=False)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)
    
    def predict(self, X, **kwargs):
        """
        Dự đoán với mô hình.
        
        Parameters:
          - X: Dữ liệu đầu vào, có dạng (batch_size, seq_len, input_dim).
          - kwargs: Các tham số khác (ví dụ: fine_tune_data, num_epochs_ft, lr_ft, batch_size).
        
        Nếu fine_tune_data được cung cấp, mô hình sẽ được fine-tune trước khi dự đoán.
        """
        fine_tune_data = kwargs.get("fine_tune_data", None)
        num_epochs_ft = kwargs.get("num_epochs_ft", 5)
        lr_ft = kwargs.get("lr_ft", 1e-4)
        batch_size = kwargs.get("batch_size", 32)
        
        if fine_tune_data is not None:
            X_ft, y_ft = fine_tune_data
            self.train_model(X_ft, y_ft, loss_fn=nn.MSELoss(), num_epochs=num_epochs_ft, lr=lr_ft, batch_size=batch_size)
        
        self.eval()
        with torch.no_grad():
            predictions = self(torch.tensor(X, dtype=torch.float32))
        return predictions.numpy()
