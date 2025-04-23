import numpy as np 
from data.process import StockDataProcessor
from model.common import BaseModel, BasePytorchModel
from model.common import mse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
def inverse_y(y, data_processor: StockDataProcessor):
    ndays = y.shape[1]
    y_new = np.zeros(y.shape)
    for i in range(ndays):
        y_new[:, i] = data_processor.inverse_transform(y[:, i], 'Close')[:, 0]
    return y_new

class TestSet:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, data_processor: StockDataProcessor):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.data_processor = data_processor
        
def _predict(model: BaseModel, test_set: TestSet, ndays: int):
    # Single predict time
    start_time = time.time()
    y_train_pred = model.predict(test_set.X_train[0:1], n_days=ndays)
    end_time = time.time()
    single_predict_time = end_time - start_time
    

    y_train_pred = model.predict(test_set.X_train, n_days=ndays)
    y_val_pred = model.predict(test_set.X_val, n_days=ndays)
    y_test_pred = model.predict(test_set.X_test, n_days=ndays)
    
    y_train_scaled = inverse_y(y_train_pred, test_set.data_processor)
    y_val_scaled = inverse_y(y_val_pred, test_set.data_processor)
    y_test_scaled = inverse_y(y_test_pred, test_set.data_processor)
    
    y_train_pred = inverse_y(y_train_pred, test_set.data_processor)
    y_val_pred = inverse_y(y_val_pred, test_set.data_processor)
    y_test_pred = inverse_y(y_test_pred, test_set.data_processor)
    
    train_mse = model.evaluate_model(y_train_scaled, y_train_pred, mse)
    val_mse = model.evaluate_model(y_val_scaled, y_val_pred, mse)
    test_mse = model.evaluate_model(y_test_scaled, y_test_pred, mse)
    
    train_da = model.evaluate_da(y_train_scaled, y_train_pred, test_set.X_train)
    val_da = model.evaluate_da(y_val_scaled, y_val_pred, test_set.X_val)
    test_da = model.evaluate_da(y_test_scaled, y_test_pred, test_set.X_test)
    
    return train_mse, val_mse, test_mse, train_da, val_da, test_da, single_predict_time
        
def test_base_model(model: BaseModel, test_set: TestSet, attempts: int=3, ndays=5):
    results = []
    for i in range(attempts):
        start_time = time.time()
        model.train_model(test_set.X_train, test_set.y_train)
        end_time = time.time()
        train_time = end_time - start_time
        
        train_mse, val_mse, test_mse, train_da, val_da, test_da, single_predict_time = _predict(model, test_set, ndays)
        
        results.append({
            'attempt': i,
            'model_name': model.name,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_da': train_da,
            'val_da': val_da,
            'test_da': test_da,
            'train_time': train_time,
            'single_predict_time': single_predict_time
        })
        
    return results

def set_seed(seed: int, use_kaggle: bool=True):
    if use_kaggle:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def test_pytorch_model(model: BasePytorchModel, test_set: TestSet, random_seed: int, 
                       lr: float, best_model_path: str, attempts: int=3, ndays: int=5):
    results = []
    for i in range(attempts):
        set_seed(random_seed)
        start_time = time.time()
        model.train_model(X=test_set.X_train, y=test_set.y_train, loss_fn=nn.MSELoss(), num_epochs=40, lr=lr, batch_size=32, 
                  X_val=test_set.X_val, y_val=test_set.y_val, use_warmup=True, warmup_epochs=6, 
                  scheduler_type="cosine", scheduler_params={"T_max": 35, "eta_min": 1e-6},
                  optimizer_class=optim.Adam, optimizer_params={"betas": (0.9, 0.999), "eps": 1e-8},
                  early_stopping=True, patience=20, save_best_model=True, best_model_checkpoint=best_model_path,
                  force_override_optimizer=True, debug=True)
        end_time = time.time()
        train_time = end_time - start_time
        
        # Use last epoch
        train_mse, val_mse, test_mse, train_da, val_da, test_da, single_predict_time = _predict(model, test_set, ndays)
        
        results.append({
            'attempt': i,
            'model_name': model.name,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_da': train_da,
            'val_da': val_da,
            'test_da': test_da,
            'train_time': train_time,
            'single_predict_time': single_predict_time,
            'type': 'last'
        })
        
        # Use best epoch
        model.load_model(best_model_path, set_eval=True)
        train_mse, val_mse, test_mse, train_da, val_da, test_da, single_predict_time = _predict(model, test_set, ndays)
        
        results.append({
            'attempt': i,
            'model_name': model.name,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_da': train_da,
            'val_da': val_da,
            'test_da': test_da, 
            'train_time': train_time,
            'single_predict_time': single_predict_time,
            'type': 'best'
        })
        
    return results
