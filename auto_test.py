import numpy as np 
from data.process import StockDataProcessor
from model.common import BaseModel, BasePytorchModel
from model.common import mse, mae, corr, supported_metrics
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

class TrainingSet:
    def __init__(self, X_train, y_train, X_val, y_val, data_processor: StockDataProcessor):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.data_processor = data_processor
        
class TestingSet:
    def __init__(self, X_tests: list, y_tests: list, data_processor: StockDataProcessor):
        self.X_tests = X_tests
        self.y_tests = y_tests
        self.data_processor = data_processor
        
def set_seed(seed, use_kaggle: bool=True):
    if use_kaggle:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def predict_and_evaluate(model: BaseModel, X, y, data_processor: StockDataProcessor,
                         metric: supported_metrics|None=None,
                         is_da: bool=False, ndays: int=5):
    y_pred = model.predict(X, ndays=ndays)
    
    y_true_rev = inverse_y(y, data_processor)
    y_pred_rev = inverse_y(y_pred, data_processor)
    
    if is_da is False:
        perdays, mean = model.evaluate_model(y_true_rev, y_pred_rev, metric)
    else:
        y_prev_scaled = X[:, -1, 0:1]  # vẫn giữ shape (L, 1) để inverse_transform
        y_prev_inv = data_processor.inverse_transform(y_prev_scaled, 'Close')[:, 0]  # lấy lại giá thật
        perdays, mean = model.evaluate_da(y_true_rev, y_pred_rev, y_prev_inv)
    return perdays, mean

def predict_and_evaluate_all(model: BaseModel, X, y, data_processor: StockDataProcessor, ndays: int=5):
    metrics = [mse, mae, corr]
    result = {
        'mse': {},
        'mae': {},
        'corr': {},
        'da': {}
    }
    
    for metric in metrics:
        perdays, mean = predict_and_evaluate(model, X, y, data_processor, metric, ndays=ndays)
        result[metric.name] = {
            'first': perdays[0],
            'last': perdays[-1],
            'mean': mean
        }
    
    perdays, mean = predict_and_evaluate(model, X, y, data_processor, is_da=True, ndays=ndays)
    result['da'] = {
        'first': perdays[0],
        'last': perdays[-1],
        'mean': mean
    }
    
    return result
    
def test_base_model(model: BaseModel, training_set: TrainingSet, testing_set: TestingSet,
                    data_processor: StockDataProcessor, attempts: int=3, ndays: int=5):
    result = {
        'model': model.name,
        'num_attempts': attempts,
        'attempts': []
    }
    
    for i in range(attempts):
        new_attempt_result = {
            'id': i,
            'train_time': 0,
            'single_predict_time': 0,
            'train': {},
            'valid': {},
            'tests': []
        }
        
        # Train phase
        start_time = time.time()
        model.train_model(training_set.X_train, training_set.y_train)
        train_time = time.time() - start_time
        new_attempt_result['train_time'] = train_time
        
        # Single predict phase
        start_time = time.time()
        y_pred = model.predict(training_set.X_train[0:1], ndays=ndays)
        single_predict_time = time.time() - start_time
        new_attempt_result['single_predict_time'] = single_predict_time
        
        # Train
        new_attempt_result['train'] = predict_and_evaluate_all(model, training_set.X_train, training_set.y_train, data_processor, ndays)
        
        # Valid
        new_attempt_result['valid'] = predict_and_evaluate_all(model, training_set.X_val, training_set.y_val, data_processor, ndays)
        
        # Test
        for j in range(len(testing_set.X_tests)):
            X_test = testing_set.X_tests[j]
            y_test = testing_set.y_tests[j]
            evaluated_result = predict_and_evaluate_all(model, X_test, y_test, data_processor, ndays)
            evaluated_result['test_id'] = j
            new_attempt_result['tests'].append(evaluated_result)
            
        result['attempts'].append(new_attempt_result)
        
    return result

def test_pytorch_model(model: BasePytorchModel, training_set: TrainingSet, testing_set: TestingSet,
                       data_processor: StockDataProcessor, random_seed: int, lr: float,
                       best_model_path: str, attempts: int=3, ndays: int=5):
    result = {
        'model': model.name,
        'num_attempts': attempts,
        'attempts': []
    }
    
    model.save_init_state()
    
    for i in range(attempts):
        new_attempt_result = {
            'id': i,
            'train_time': 0,
            'single_predict_time': 0,
            'train': [],
            'valid': [],
            'tests': []
        }
        
        # Train phase
        start_time = time.time()
        set_seed(random_seed)
        model.reset_state()
        model.train_model(
            X=training_set.X_train, y=training_set.y_train,
            loss_fn=nn.MSELoss(), num_epochs=40, lr=lr, batch_size=32,
            X_val=training_set.X_val, y_val=training_set.y_val,
            use_warmup=True, warmup_epochs=6,
            scheduler_type="cosine", scheduler_params={"T_max": 35, "eta_min": 1e-6},
            optimizer_class=optim.Adam, optimizer_params={"betas": (0.9,0.999), "eps":1e-8},
            early_stopping=True, patience=20,
            save_best_model=True, best_model_checkpoint=best_model_path,
            force_override_optimizer=True, debug=False
        )
        train_time = time.time() - start_time
        new_attempt_result['train_time'] = train_time
        
        # Single predict phase
        start_time = time.time()
        y_pred = model.predict(training_set.X_train[0:1], ndays=ndays)
        single_predict_time = time.time() - start_time
        new_attempt_result['single_predict_time'] = single_predict_time
        
        for type in ['last', 'best']:
            if type == 'best':
                model.load_model(best_model_path, set_eval=True, debug=False)
            
            # Train
            train_result = predict_and_evaluate_all(model, training_set.X_train, training_set.y_train, data_processor, ndays)
            train_result['type'] = type
            new_attempt_result['train'].append(train_result)
            
            # Valid
            valid_result = predict_and_evaluate_all(model, training_set.X_val, training_set.y_val, data_processor, ndays)
            valid_result['type'] = type
            new_attempt_result['valid'].append(valid_result)
            
            # Test
            test_result = {
                'type': type,
                'tests': []
            }
            for j in range(len(testing_set.X_tests)):
                X_test = testing_set.X_tests[j]
                y_test = testing_set.y_tests[j]
                evaluated_result = predict_and_evaluate_all(model, X_test, y_test, data_processor, ndays)
                evaluated_result['test_id'] = j
                test_result['tests'].append(evaluated_result)
                
            new_attempt_result['tests'].append(test_result)
                
        result['attempts'].append(new_attempt_result)
        
    return result