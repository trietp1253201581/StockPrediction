import numpy as np
from model.common import BaseModel

class BaselineLastDayModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        
    def train_model(self, X, y, **kwargs):
        print('Model does not need to train')
    
    def predict(self, X, ndays = 5):
        # Lấy giá của ngày cuối cùng (ở cột 0, giả sử đây là 'Close')
        last_day = X[:, -1, 0]  # shape (L,)
        # Tạo mảng với shape (L, ndays) bằng cách lặp lại giá này
        return np.repeat(last_day[:, np.newaxis], ndays, axis=1)
    
class BaselineMAModel(BaseModel):
    def __init__(self, ma_window: int = 10):
        super().__init__()
        self.ma_window = 10
        
    def train_model(self, X, y, **kwargs):
        print('Model does not need to train')
    
    def predict(self, X, ndays = 5):
        L = X.shape[0]
        preds = np.zeros((L, ndays))
        
        # Với mỗi mẫu riêng biệt
        for i in range(L):
            # Lấy cửa sổ ban đầu từ X: sử dụng self.ma_window giá cuối cùng từ cột 0
            current_window = X[i, -self.ma_window:, 0].copy()  # shape (ma_window,)
            for d in range(ndays):
                pred_day = np.mean(current_window)
                preds[i, d] = pred_day
                # Cập nhật cửa sổ: loại bỏ giá cũ nhất, thêm giá dự đoán vừa tính
                current_window = np.append(current_window[1:], pred_day)
        return preds