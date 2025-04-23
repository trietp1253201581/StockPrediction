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

class TrainingSet:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, data_processor: StockDataProcessor):
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
        
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------------------------------------------

def inverse_y(y, data_processor):
    # Chuyển scale trở về giá trị gốc
    ndays = y.shape[1]
    y_new = np.zeros_like(y)
    for i in range(ndays):
        y_new[:, i] = data_processor.inverse_transform(y[:, i], 'Close')[:, 0]
    return y_new

def direction_accuracy(y_true, y_pred, y_prev):
    delta_true = y_true - y_prev
    delta_pred = y_pred - y_prev
    return np.mean(np.sign(delta_true) == np.sign(delta_pred))

# ------------------------------------------------------------------------------------------------
# BaseModel testing
# ------------------------------------------------------------------------------------------------

def test_base_model(model: BaseModel,
                    training_set: TrainingSet,
                    testing_set: TestingSet,
                    attempts: int = 3,
                    ndays: int = 5):
    """
    Test BaseModel (non-PyTorch) trên training_set (1 train/val)
    và testing_set (danh sách nhiều X_test/y_test).
    Trả về list of dict với các key:
      - run, model, type='last'
      - train_time
      - train_mse_perday, train_mse_mean, train_da_perday, train_da_mean
      - val_..., test_mse_mean_mean, test_mse_mean_std, test_da_mean_mean, test_da_mean_std
    """
    results = []
    for run in range(attempts):
        # 1) Train
        t0 = time.time()
        model.train_model(training_set.X_train, training_set.y_train)
        train_time = time.time() - t0

        # 2) Eval train & val
        # predict & inverse
        y_tr_pred = model.predict(training_set.X_train, ndays=ndays)
        y_vl_pred = model.predict(training_set.X_val,   ndays=ndays)
        y_tr_true = inverse_y(y_tr_pred, training_set.data_processor)
        y_vl_true = inverse_y(y_vl_pred, training_set.data_processor)
        y_tr_pred = inverse_y(y_tr_pred, training_set.data_processor)
        y_vl_pred = inverse_y(y_vl_pred, training_set.data_processor)

        # metrics
        train_mse_pd, train_mse_m = model.evaluate_model(y_tr_true, y_tr_pred, mse)
        val_mse_pd,   val_mse_m   = model.evaluate_model(y_vl_true, y_vl_pred, mse)

        y_prev_tr = training_set.X_train[:, -1, 0]
        y_prev_vl = training_set.X_val[:,   -1, 0]
        train_da_pd = np.array([direction_accuracy(y_tr_true[:,i], y_tr_pred[:,i], y_prev_tr)
                                 for i in range(ndays)])
        val_da_pd   = np.array([direction_accuracy(y_vl_true[:,i], y_vl_pred[:,i], y_prev_vl)
                                 for i in range(ndays)])
        train_da_m = train_da_pd.mean()
        val_da_m   = val_da_pd.mean()

        # 3) Eval test splits
        test_mse_means, test_da_means = [], []
        for X_t, y_t in zip(testing_set.X_tests, testing_set.y_tests):
            y_t_pred = model.predict(X_t, ndays=ndays)
            y_t_true = inverse_y(y_t_pred, testing_set.data_processor)
            y_t_pred = inverse_y(y_t_pred, testing_set.data_processor)
            mse_pd, mse_m = model.evaluate_model(y_t_true, y_t_pred, mse)

            y_prev_t = X_t[:, -1, 0]
            da_pd   = np.array([direction_accuracy(y_t_true[:,i], y_t_pred[:,i], y_prev_t)
                                 for i in range(ndays)])
            da_m    = da_pd.mean()

            test_mse_means.append(mse_m)
            test_da_means.append(da_m)

        test_summary = {
            'test_mse_mean_mean': np.mean(test_mse_means),
            'test_mse_mean_std':  np.std(test_mse_means),
            'test_da_mean_mean':  np.mean(test_da_means),
            'test_da_mean_std':   np.std(test_da_means)
        }

        # 4) Append result
        results.append({
            'run': run,
            'model': model.name,
            'type': 'last',
            'train_time': train_time,
            # train
            'train_mse_perday': train_mse_pd,
            'train_mse_mean':   train_mse_m,
            'train_da_perday':  train_da_pd,
            'train_da_mean':    train_da_m,
            # val
            'val_mse_perday':   val_mse_pd,
            'val_mse_mean':     val_mse_m,
            'val_da_perday':    val_da_pd,
            'val_da_mean':      val_da_m,
            # test
            **test_summary
        })
    from datetime import datetime
    datetime()
    return results

# ------------------------------------------------------------------------------------------------
# BasePytorchModel testing
# ------------------------------------------------------------------------------------------------

def test_pytorch_model(model: BasePytorchModel,
                       training_set: TrainingSet,
                       testing_set: TestingSet,
                       random_seed: int,
                       lr: float,
                       best_model_path: str,
                       attempts: int = 3,
                       ndays: int = 5):
    """
    Tương tự test_base_model nhưng dùng train_model có scheduler, checkpoint, warmup...
    Trả về list of dict với thêm type='last' và type='best' nếu checkpoint được load.
    """
    results = []
    for run in range(attempts):
        # set seed
        torch.manual_seed(random_seed + run)
        # 1) train
        t0 = time.time()
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
        train_time = time.time() - t0

        # define helper to eval one split
        def eval_split(X, y, prev_source):
            y_pred = model.predict(X, ndays=ndays)
            y_true = inverse_y(y_pred, testing_set.data_processor)
            y_pred_s = inverse_y(y_pred, testing_set.data_processor)

            mse_pd, mse_m = model.evaluate_model(y_true, y_pred_s, mse)
            y_prev = prev_source[:, -1, 0]
            da_pd = np.array([direction_accuracy(y_true[:,i], y_pred_s[:,i], y_prev)
                               for i in range(ndays)])
            da_m = da_pd.mean()
            return mse_pd, mse_m, da_pd, da_m

        # 2) train/val
        tr = eval_split(training_set.X_train, training_set.y_train, training_set.X_train)
        vl = eval_split(training_set.X_val,   training_set.y_val,   training_set.X_val)

        # 3) test splits
        test_mse_means, test_da_means = [], []
        for X_t, y_t in zip(testing_set.X_tests, testing_set.y_tests):
            _, m_m, _, d_m = eval_split(X_t, y_t, X_t)
            test_mse_means.append(m_m)
            test_da_means.append(d_m)
        test_summary = {
            'test_mse_mean_mean': np.mean(test_mse_means),
            'test_mse_mean_std':  np.std(test_mse_means),
            'test_da_mean_mean':  np.mean(test_da_means),
            'test_da_mean_std':   np.std(test_da_means)
        }

        # 4) append last
        results.append({
            'run': run,
            'model': model.name,
            'type': 'last',
            'train_time': train_time,
            'train_mse_perday': tr[0],
            'train_mse_mean':   tr[1],
            'train_da_perday':  tr[2],
            'train_da_mean':    tr[3],
            'val_mse_perday':   vl[0],
            'val_mse_mean':     vl[1],
            'val_da_perday':    vl[2],
            'val_da_mean':      vl[3],
            **test_summary
        })

        # 5) append best (nếu có)
        model.load_model(best_model_path, set_eval=True)
        tr = eval_split(training_set.X_train, training_set.y_train, training_set.X_train)
        vl = eval_split(training_set.X_val,   training_set.y_val,   training_set.X_val)
        test_mse_means, test_da_means = [], []
        for X_t, y_t in zip(testing_set.X_tests, testing_set.y_tests):
            _, m_m, _, d_m = eval_split(X_t, y_t, X_t)
            test_mse_means.append(m_m)
            test_da_means.append(d_m)
        test_summary = {
            'test_mse_mean_mean': np.mean(test_mse_means),
            'test_mse_mean_std':  np.std(test_mse_means),
            'test_da_mean_mean':  np.mean(test_da_means),
            'test_da_mean_std':   np.std(test_da_means)
        }

        results.append({
            'run': run,
            'model': model.name,
            'type': 'best',
            'train_time': train_time,
            'train_mse_perday': tr[0],
            'train_mse_mean':   tr[1],
            'train_da_perday':  tr[2],
            'train_da_mean':    tr[3],
            'val_mse_perday':   vl[0],
            'val_mse_mean':     vl[1],
            'val_da_perday':    vl[2],
            'val_da_mean':      vl[3],
            **test_summary
        })

    return results
