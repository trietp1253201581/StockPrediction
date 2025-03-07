import yfinance as yf
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split


def extract_stock_data(tickers: list[str],
                       storage_path: str,
                       start_date: datetime,
                       end_date: datetime):
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    data = yf.download(tickers=tickers, start=start, end=end, group_by='ticker')
    data.to_csv(storage_path)
    
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime

class StockDataProcessor:
    def __init__(self):
        self.df = None
        self.scalers = {}  # Dictionary lưu scaler cho từng feature

    def load_raw_to_df(self, raw_data_storage_path: str, tickers: list[str]):
        df = pd.read_csv(raw_data_storage_path)
        df.drop(index=[0, 1], inplace=True)
        new_data = {
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': [],
            'Ticker': [],
            'Collect Date': []
        }
        col_lst = df.columns.tolist()
        for i in range(0, len(tickers)):
            this_ticker = col_lst[5 * i + 1]
            new_data['Open'].extend(df[col_lst[5 * i + 1]].values)
            new_data['High'].extend(df[col_lst[5 * i + 2]].values)
            new_data['Low'].extend(df[col_lst[5 * i + 3]].values)
            new_data['Close'].extend(df[col_lst[5 * i + 4]].values)
            new_data['Volume'].extend(df[col_lst[5 * i + 5]].values)
            new_data['Ticker'].extend([this_ticker] * len(df))
            new_data['Collect Date'].extend(df['Ticker'])  # Giả sử cột 'Ticker' ở đây chứa ngày thu thập
        new_df = pd.DataFrame(new_data)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            new_df[col] = new_df[col].astype('float32')
        new_df['Collect Date'] = new_df['Collect Date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
        self.df = new_df.copy()
        del df, new_df

    def handle_missing_data(self, start_date: datetime, end_date: datetime, features: list[str]):
        self.df[features] = self.df.groupby('Ticker')[features].ffill().bfill()
        self.df = self.df[(self.df['Collect Date'] >= start_date) & (self.df['Collect Date'] <= end_date)].copy()
        
    def scale(self, features: list[str], scaler_class=RobustScaler):
        """
        Fit scaler cho từng feature và cập nhật dữ liệu trong self.df.
        Lưu lại các scaler trong self.scalers dưới dạng dictionary.
        """
        self.scalers = {}
        for feature in features:
            scaler = scaler_class()
            # Fit scaler trên cột feature (chú ý chuyển DataFrame sang mảng 2D)
            self.df[feature] = scaler.fit_transform(self.df[[feature]])
            self.scalers[feature] = scaler
        
    def select_feature(self, features: list[str], date_col='Collect Date', ticker_col='Ticker'):
        self.df = self.df[features + [date_col, ticker_col]]
        
    @staticmethod
    def _create_sliding_window(df_ticker: pd.DataFrame, window_for_x: int, window_for_y: int, 
                               target_col: str, features: list[str]):
        """
        Tạo các mẫu sliding window cho một ticker.
        - X: (num_samples, window_for_x, len(features))
        - y: (num_samples, window_for_y), y là các giá trị của target_col từ
             i+window_for_x đến i+window_for_x+window_for_y-1.
        """
        data = df_ticker[features].values
        X, y = [], []
        total_window = window_for_x + window_for_y
        target_index = features.index(target_col)
        for i in range(len(data) - total_window + 1):
            X.append(data[i : i + window_for_x])
            y_window = data[i + window_for_x : i + total_window, target_index]
            y.append(y_window)
        return np.array(X), np.array(y)
    
    def split_train_val_test(self, window_for_x=50, window_for_y=5, val_size=0.1, test_size=0.1, 
                             features: list[str] = None, target_col: str = None):
        """
        Chia dữ liệu thành train, validation và test theo từng ticker trước khi tạo sliding window.
        Mỗi mẫu X có shape (window_for_x, len(features)), y có shape (window_for_y,).
        """
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []

        # Nhóm theo ticker
        for ticker, group in self.df.groupby('Ticker'):
            group_sorted = group.sort_values('Collect Date')
            n = len(group_sorted)
            n_val = int(n * val_size)
            n_test = int(n * test_size)
            n_train = n - n_val - n_test

            train_group = group_sorted.iloc[:n_train]
            val_group = group_sorted.iloc[n_train:n_train + n_val]
            test_group = group_sorted.iloc[n_train + n_val:]

            total_needed = window_for_x + window_for_y

            if len(train_group) >= total_needed:
                X_train, y_train = StockDataProcessor._create_sliding_window(train_group, window_for_x, window_for_y, target_col, features)
                X_train_list.append(X_train)
                y_train_list.append(y_train)

            if len(val_group) >= total_needed:
                X_val, y_val = StockDataProcessor._create_sliding_window(val_group, window_for_x, window_for_y, target_col, features)
                X_val_list.append(X_val)
                y_val_list.append(y_val)

            if len(test_group) >= total_needed:
                X_test, y_test = StockDataProcessor._create_sliding_window(test_group, window_for_x, window_for_y, target_col, features)
                X_test_list.append(X_test)
                y_test_list.append(y_test)

        X_train_all = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
        y_train_all = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
        
        X_val_all = np.concatenate(X_val_list, axis=0) if X_val_list else np.array([])
        y_val_all = np.concatenate(y_val_list, axis=0) if y_val_list else np.array([])
        
        X_test_all = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
        y_test_all = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

        return X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all

    def inverse_transform(self, x: np.ndarray, feature: str):
        """
        Inverse transform giá trị dự đoán của một feature cụ thể.
        - x: np.ndarray có shape (n, 1) hoặc (n,) chứa giá trị đã scale của feature.
        - feature: Tên feature, ví dụ 'Close'
        Trả về np.ndarray với giá trị gốc.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        scaler = self.scalers.get(feature)
        if scaler is None:
            raise ValueError(f"Scaler cho feature '{feature}' không tồn tại.")
        return scaler.inverse_transform(x)

            
        