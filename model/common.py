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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_class = None
        self.optimizer_params = None
        self.optimizer = None
        self.scheduler = None  # Bộ lập lịch learning rate
        self.last_epoch = 0
        self.last_loss = 0.0
        self.trained = False
        self.to(self.device)

    def init_optimizer(self, optimizer_class: type[optim.Optimizer], optimizer_params=None, lr:float=1e-4):
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        if self.optimizer is not None:
            print("Warning: Overriding existing optimizer.")
        self.optimizer = optimizer_class(self.parameters(), lr=lr, **self.optimizer_params)

    def init_scheduler(self, scheduler_type, scheduler_params):
        if scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

    @staticmethod
    def make_loader(X, y, batch_size=32, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        return loader
    
    def save_model(self, file_path="model_checkpoint.pth"):
        if not self.trained:
            raise Exception("Model is not trained yet.")
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": self.last_epoch,
            "last_loss": self.last_loss
        }, file_path)

    def load_model(self, file_path="model_checkpoint.pth", set_eval=True):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is None:
            self.init_optimizer(lr=1e-4)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.last_epoch = checkpoint.get("last_epoch", 0)
        self.last_loss = checkpoint.get("last_loss", 0.0)
        self.eval() if set_eval else self.train()
        print(f"Model loaded from {file_path}, starting from epoch {self.last_epoch}, eval mode: {set_eval}")


    def train_model(self, X, y, loss_fn, num_epochs, lr, optimizer_class: type[optim.Optimizer], optimizer_params=None,
                    batch_size=32, X_val=None, y_val=None,
                    use_warmup=False, warmup_epochs=5, scheduler_type=None, scheduler_params=None):
        train_loader = BasePytorchModel.make_loader(X, y, batch_size=batch_size, shuffle=True)
        self.init_optimizer(optimizer_class, optimizer_params, lr)
        
        if use_warmup:
            warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        
        if scheduler_type:
            self.init_scheduler(scheduler_type, scheduler_params)

        for epoch in range(self.last_epoch + 1, self.last_epoch + num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(train_loader.dataset)
            
            if use_warmup and epoch <= warmup_epochs:
                warmup_scheduler.step()
            elif self.scheduler:
                if scheduler_type == "reduce_on_plateau":
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()
            
            if X_val is not None and y_val is not None:
                val_loss = self.valid_model(X_val, y_val, loss_fn, batch_size=batch_size*2)
                print(f"Epoch {epoch}/{num_epochs+self.last_epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch}/{num_epochs+self.last_epoch}, Train Loss: {epoch_loss:.4f}")
            self.last_epoch = epoch
            self.last_loss = epoch_loss
        self.trained = True

    def valid_model(self, X, y, loss_fn, batch_size=32):
        loader = BasePytorchModel.make_loader(X, y, batch_size=batch_size, shuffle=False)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def predict(self, X, **kwargs):
        fine_tune_data = kwargs.get("fine_tune_data", None)
        num_epochs_ft = kwargs.get("num_epochs_ft", 5)
        lr_ft = kwargs.get("lr_ft", 1e-4)
        batch_size = kwargs.get("batch_size", 64)  # Đảm bảo batch_size có giá trị hợp lý
        optimizer_class = kwargs.get("optimizer_class", self.optimizer_class)
        optimizer_params = kwargs.get("optimizer_params", self.optimizer_params)

        # Fine-tuning nếu có dữ liệu
        if fine_tune_data is not None:
            X_ft, y_ft = fine_tune_data
            self.train_model(X_ft, y_ft, loss_fn=nn.MSELoss(), num_epochs=num_epochs_ft, lr=lr_ft, 
                             optimizer_class=optimizer_class, optimizer_params=optimizer_params, batch_size=batch_size)

        self.eval()
        predictions = []
        
        with torch.no_grad():
            loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32, device=self.device)), 
                                batch_size=batch_size, shuffle=False)

            for X_batch in loader:
                X_batch = X_batch[0]  # Lấy dữ liệu từ tuple
                preds = self(X_batch)
                predictions.append(preds.cpu().numpy())
        
        return np.vstack(predictions)  # Ghép lại thành một mảng numpy đầy đủ

