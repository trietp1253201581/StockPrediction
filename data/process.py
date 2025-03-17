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
    """
    Lấy dữ liệu cổ phiếu từ API của Yahoo Finance, sử dụng
    thư viện yfinance của python

    Args:
        tickers (list[str]): Danh sách các mã cổ phiếu cần lấy
        storage_path (str): Đường dẫn tới file sẽ lưu dữ liệu lấy được
        start_date (datetime): Ngày bắt đầu lấy cổ phiếu
        end_date (datetime): Ngày cuối cùng lấy cổ phiếu
    """
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    data = yf.download(tickers=tickers, start=start, end=end, group_by='ticker')
    data.to_csv(storage_path)

class StockDataProcessor:
    """
    Lớp trợ giúp xử lý dữ liệu cổ phiếu thô
    """
    def __init__(self):
        self.df = None
        self.scalers = {}  # Dictionary lưu scaler cho từng feature

    def load_raw_to_df(self, raw_data_storage_path: str, tickers: list[str]):
        """
        Đọc file dữ liệu thô và chuyển đổi định dạng để được DataFrame cho các bước
        xử lý tiếp theo.
        
        Các format lại sẽ bao gồm:
        - Đưa dữ liệu thô về dạng cột đơn
        - Thêm cột Ticker và Collect Date
        - Chuyển đổi kiểu dữ liệu thống nhất là float 32

        Args:
            raw_data_storage_path (str): Đường dẫn tới file lưu trữ dữ liệu thô
            tickers (list[str]): Danh sách các ticker được giữ lại
        """
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
        """
        Xử lý dữ liệu thiếu trên các cột bằng ffill và bfill (dữ liệu trước gần nhất, dữ liệu sau gần nhất).
        Chỉ những ngày nằm trong khoảng từ `start_date` tới `end_date` mới được xử lý dữ liệu thiếu.

        Args:
            start_date (datetime): Ngày bắt đầu điền dữ liệu thiếu
            end_date (datetime): Ngày cuối cùng điền dữ liệu thiếu
            features (list[str]): Các đặc trưng cần điền dữ liệu thiếu
        """
        self.df[features] = self.df.groupby('Ticker')[features].ffill().bfill()
        self.df = self.df[(self.df['Collect Date'] >= start_date) & (self.df['Collect Date'] <= end_date)].copy()
        
    def scale(self, features: list[str], scaler_class=RobustScaler):
        """
        Scale lại các dữ liệu cần thiết, các scaler được sử dụng sẽ được lưu lại
        phục vụ cho việc inverse sau này.
        
        Mỗi feature sẽ sử dụng một scaler khác nhau

        Args:
            features (list[str]): Danh sách các đặc trưng cần scale
            scaler_class (_type_, optional): Scaler được sử dụng, phải chứa các phương thức
                `fit`, `transform`, `fit_transform`, `inverse_transform`.
                Nên sử dụng các Scaler từ `sklearn`. Defaults to RobustScaler.
        """
        self.scalers = {}
        for feature in features:
            scaler = scaler_class()
            # Fit scaler trên cột feature (chú ý chuyển DataFrame sang mảng 2D)
            self.df[feature] = scaler.fit_transform(self.df[[feature]])
            self.scalers[feature] = scaler
        
    def select_feature(self, features: list[str], date_col='Collect Date', ticker_col='Ticker'):
        """
        Giảm số chiều đầu vào bằng cách lựa chọn đặc trưng

        Args:
            features (list[str]): Các đặc trưng dữ lại
            date_col (str, optional): Cột biểu diễn ngày mà dữ liệu thuộc về. Defaults to 'Collect Date'.
            ticker_col (str, optional): Cột biểu diễn loại cổ phiếu. Defaults to 'Ticker'.
        """
        self.df = self.df[features + [date_col, ticker_col]]
        
    @staticmethod
    def _create_sliding_window(df_ticker: pd.DataFrame, window_for_x: int, window_for_y: int, 
                               target_col: str, features: list[str]):
        """
        Tạo một mẫu slide window cho một dữ liệu (format lại dữ liệu cổ phiếu các ngày bằng 
        cách thêm vào dữ liệu các ngày trước đó vào cùng 1 đầu vào)

        Args:
            df_ticker (pd.DataFrame): DataFrame chứa dữ liệu đầu vào theo từng ngày, từng cổ phiếu
            window_for_x (int): Cửa số cho đầu vào, đây chính là số ngày trong quá khứ dùng để dự đoán.
            window_for_y (int): Cửa số cho đầu ra, đây chính là số ngày trong tương lai cần dự đoán.
            target_col (str): Cột đích cần dự đoán
            features (list[str]): Danh sách các đặc trưng cần tạo slide window

        Returns:
            tuple: Một tuple dạng (X, y). Trong đó X là vector đầu vào kích thước (L, window_for_x),
                y là một vector đầu ra có kích thước (L, window_for_y).
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
                             features: list[str]|None = None, target_col: str|None = None):
        """
        Chia tập dữ liệu hiện tại thành 3 tập train-validation-test với tỉ lệ xác định.
        Các tập dữ liệu này biểu diễn dưới các cặp (X,y) trong đó X,y đều là các vector
        dạng slide window.

        Args:
            window_for_x (int, optional): Cửa số cho đầu vào, đây chính là số ngày trong quá khứ dùng để dự đoán. 
                Defaults to 50.
            window_for_y (int, optional): Cửa số cho đầu ra, đây chính là số ngày trong tương lai cần dự đoán.
                Defaults to 5.
            val_size (float, optional): Tỉ lệ dùng cho tập validation. Defaults to 0.1.
            test_size (float, optional): Tỉ lệ dùng cho tập test. Defaults to 0.1.
            features (list[str] | None, optional): Danh sách các đặc trưng được chọn. Defaults to None.
            target_col (str | None, optional): Cột đích cần dự đoán. Defaults to None.

        Returns:
            tuple: tuple gồm 6 phần tử dạng (X_train, y_train, X_val, y_val, X_test, y_test).
                Các vector X đều có shape(Li, window_for_x), các vector y đều có shape (Li, window_for_y)
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
        Lấy giá trị trước khi scale của một đặc trưng cụ thể

        Args:
            x (np.ndarray): Danh sách các giá trị của đặc trưng cần chuyển đổi lại
            feature (str): Đặc trưng cần inverse
            
        Raises:
            ValueError: Khi Scaler cho đặc trưng này không tồn tại

        Returns:
            ndarray: Danh sách các giá trị sau khi được inverse
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        scaler = self.scalers.get(feature)
        if scaler is None:
            raise ValueError(f"Scaler cho feature '{feature}' không tồn tại.")
        return scaler.inverse_transform(x)

            
        