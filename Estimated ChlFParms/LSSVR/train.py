#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :train.py
@Description  :
@Time         :2024/06/26 15:38:07
@Author       :Tangh
@Version      :1.0
'''

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from lssvr import LSSVR
from LoadDataset_Params_Raw import Fusion_OJIP_Dataset
import logging
from torch.utils.data import DataLoader, ConcatDataset

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")

def load_datasets(file_path_l, file_path_d, batch_size=16):
    train_dataset = Fusion_OJIP_Dataset(file_path_l, file_path_d, train=True, val=False, train_ratio=0.8, val_ratio=0.1)
    val_dataset = Fusion_OJIP_Dataset(file_path_l, file_path_d, train=False, val=True, train_ratio=0.8, val_ratio=0.1)
    test_dataset = Fusion_OJIP_Dataset(file_path_l, file_path_d, train=False, val=False, train_ratio=0.8, val_ratio=0.1)
    
    # 合并训练集和验证集
    combined_train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # 创建 DataLoader
    train_loader = DataLoader(combined_train_val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    
    return train_loader, test_loader

def train_model(train_loader):
    model = MultiOutputRegressor(LSSVR())
    for X_train, y_train in train_loader:
        X_train = to_numpy(X_train)
        y_train = to_numpy(y_train)
        X_train = X_train.squeeze()
        print(X_train.shape,y_train.shape)
        model.fit(X_train, y_train)
    return model

def evaluate_model(model, test_loader):
    y_true, y_pred = [], []
    for X_test, y_test in test_loader:
        X_test = to_numpy(X_test)
        y_test = to_numpy(y_test)
        X_test = X_test.squeeze()
        y_pred_chunk = model.predict(X_test)
        y_true.append(y_test)
        y_pred.append(y_pred_chunk)
    
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    
    # 反归一化
    y_pred = y_pred * test_loader.dataset.new_std_d + test_loader.dataset.new_mean_d
    y_true = y_true * test_loader.dataset.new_std_d + test_loader.dataset.new_mean_d
    
    r2_res = 0
    rmspe = 0
    rpd = 0
    smape = 0
    sample_length = y_true.shape[1]
    
    print('sample_length:', sample_length)
    
    for i in range(sample_length):
        r2_res += r2_score(y_true[:, i], y_pred[:, i])
        rmspe += calculate_rmse(y_true[:, i], y_pred[:, i])
        rpd += calculate_rpd(y_true[:, i], y_pred[:, i])
        smape += calculate_smape(y_true[:, i], y_pred[:, i])
     
    r2_val = r2_res / sample_length
    rmspe = rmspe / sample_length
    rpd = rpd / sample_length
    smape = smape / sample_length
    
    print(y_true.shape, y_pred.shape)
    return rmspe, r2_val, rpd, smape, y_true, y_pred

def save_results(y_true, y_pred, filename="./out/test_output-811-1channel.xlsx"):
    data_list = [np.concatenate((y_t, y_p), axis=0) for y_t, y_p in zip(y_true, y_pred)]
    df = pd.DataFrame(data_list)
    df.to_excel(filename, index=False)

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error (RMSE) using sklearn"""
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_rmspe(actual, predicted):
    """Calculate Root Mean Squared Percentage Error (RMSPE)"""
    return np.sqrt(np.mean(((actual - predicted) / actual) ** 2))

def calculate_rpd(actual, predicted):
    """Calculate Relative Prediction Deviation (RPD)"""
    rmse = calculate_rmse(actual, predicted)
    sd = np.std(actual) 
    return sd / rmse

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    ape = np.abs((actual - predicted) / actual) 
    return np.mean(ape)

def calculate_smape(actual, predicted):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)"""
    actual = np.array(actual)
    forecast = np.array(predicted)
    return np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def main():
    file_path_l = './data/ChlFData_l.pkl'
    file_path_d = './data/ChlFData_d.pkl'
    
    logging.info("加载数据集...")
    train_loader, test_loader = load_datasets(file_path_l, file_path_d, batch_size=16)
    
    logging.info("训练模型...")
    model = train_model(train_loader)
    
    logging.info("评估模型...")
    rmspe_test, r2_test, rpd_test, smape_test, y_test, y_pred_test = evaluate_model(model, test_loader)
    
    logging.info(f'测试集 R2: {r2_test}')
    logging.info(f'测试集 RMSPE: {rmspe_test}')
    logging.info(f'测试集 RPD: {rpd_test}')
    logging.info(f'测试集 SMAPE: {smape_test}')
    
    logging.info("将结果保存到 Excel 文件...")
    print(y_test.shape,y_pred_test.shape)
    save_results(y_test, y_pred_test)

if __name__ == "__main__":
    main()
