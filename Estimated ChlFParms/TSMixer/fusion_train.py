#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :fusion_train.py
@Description  :
@Time         :2024/05/15 11:25:41
@Author       :Tangh
@Version      :1.0
'''

from __future__ import division

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

from TSMixer import Model  

from LoadDataset_Params_Raw import Fusion_OJIP_Dataset
from myutils import AverageMeter, initialize_logger, save_checkpoint, record_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用GPU的ID

parser = argparse.ArgumentParser(description='Image Classification:')  # 创建参数解析器对象
parser.add_argument('--model_path', type=str, default='ChlF_Model',
                    help="Set model storage path")  # 添加参数

args = parser.parse_args()  # 解析命令行参数
file_path_l = './data/ChlFData_l.pkl'
file_path_d = './data/ChlFData_d.pkl'
train_dataset = Fusion_OJIP_Dataset(file_path_l, file_path_d, train=True, val=False, train_ratio=0.6, val_ratio=0.2)
val_dataset = Fusion_OJIP_Dataset(file_path_l, file_path_d, train=False, val=True, train_ratio=0.6, val_ratio=0.2)
batch_size = 16

# 修改模型输入输出维度
seq_len = 458
pred_len = 478


# 配置参数类
class Configs:
    def __init__(self):
        self.task_name = 'short_term_forecast'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 't'
        self.output_attention = True
        self.factor = 1
        self.label_len = 48
        self.down_sampling_window = 1
        self.channel_independence = 1
        self.down_sampling_layers = 0
        self.decomp_method = 'moving_avg'
        self.down_sampling_method = None
        self.moving_avg = 25
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.use_norm = 1

def main():
    cudnn.benchmark = True  # 优化网络运行性能
    configs = Configs()
    model = Model(configs)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device=device, dtype=torch.float)

    start_epoch = 0
    end_epoch = 201
    init_lr = 1e-4
    iteration = 0
    acc_ten = np.empty(10)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False,
                                  threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=1e-10, eps=1e-8)
    model_path = './models/' + args.model_path
    max_acc = 0
    acc_index = 0

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    for epoch in range(start_epoch + 1, end_epoch):
        torch.cuda.empty_cache()
        train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_data_loader = DataLoader(dataset=val_dataset, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
        
        start_time = time.time()
        train_loss, iteration, train_r2, train_rmse, train_rpd, train_mape = train(train_data_loader, model, criterion, optimizer, iteration, device)
        val_loss, val_r2, val_rmse, val_rpd, val_mape = validate(val_data_loader, model, criterion, device)

        if max_acc < val_r2:
            max_acc = val_r2
            save_checkpoint(model_path, epoch, iteration, model, optimizer, name='best')
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        scheduler.step(val_loss)

        if epoch % 50 == 0:
            save_checkpoint(model_path, epoch, iteration, model, optimizer, name=str(epoch) + 'epoch')

        end_time = time.time()
        epoch_time = end_time - start_time
        if acc_index < 9:
            acc_ten[acc_index] = val_r2
            print(f"Epoch [{epoch}]|Iter[{iteration}]|Time:{epoch_time:.4f}|learning rate : {lr:.9f}|Train Loss: {train_loss:.4f}|Train R²:{train_r2 * 100:.2f}%|Train RMSE:{train_rmse:.4f}|Train RPD:{train_rpd:.4f}|Train MAPE:{train_mape:.4f}|"
                  f"Val Loss:{val_loss:.9f}|Val R²:{val_r2 * 100:.2f}%|Val RMSE:{val_rmse:.4f}|Val RPD:{val_rpd:.4f}|Val MAPE:{val_mape:.4f}|")
            record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss, train_r2, val_r2, train_rmse, val_rmse, train_rpd, val_rpd, train_mape, val_mape)
            logger.info(f"Epoch [{epoch}], Iter[{iteration}], Time:{epoch_time:.4f}, learning rate : {lr:.9f}, Train Loss: {train_loss:.4f}  Train R²:{train_r2 * 100:.2f}%, "
                        f"Val Loss:{val_loss:.9f}  Val R²:{val_r2 * 100:.2f}%   Near 10 Accuracy:None")
        else:
            acc_ten[acc_index % 10] = val_r2
            print(f"Epoch [{epoch}]|Iter[{iteration}]|Time:{epoch_time:.4f}|learning rate : {lr:.9f}|Train Loss: {train_loss:.4f}|Train R²:{train_r2 * 100:.2f}%|Train RMSE:{train_rmse:.4f}|Train RPD:{train_rpd:.4f}|Train MAPE:{train_mape:.4f}|"
                  f"Val Loss:{val_loss:.9f}|Val R²:{val_r2 * 100:.2f}%|Val RMSE:{val_rmse:.4f}|Val RPD:{val_rpd:.4f}|Val MAPE:{val_mape:.4f}|")
            record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss, train_r2, val_r2, train_rmse, val_rmse, train_rpd, val_rpd, train_mape, val_mape)
            logger.info(f"Epoch [{epoch}], Iter[{iteration}], Time:{epoch_time:.4f}, learning rate : {lr:.9f}, Train Loss: {train_loss:.4f}  Train R²:{train_r2 * 100:.2f}%, "
                        f"Val Loss:{val_loss:.4f}  Val R²:{val_r2 * 100:.2f}%   Near 10 Accuracy:{acc_ten.mean() * 100:.2f}%")
        
        acc_index += 1

def train(train_data_loader, model, criterion, optimizer, iteration, device):
    losses = AverageMeter()
    model.train()
    r2_train = 0
    res1 = []
    res2 = []
    for i, (data, label, plant_names) in enumerate(train_data_loader):
        data = data.to(device=device)
        label = label.to(device=device)

        iteration = iteration + 1
        data = data.permute(0, 2, 1)
        output = model(data, None, None, None)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        with torch.no_grad():
            label = label.cpu().numpy()
            output = output.data.cpu().numpy()
            for j in range(batch_size):
                res1.append(label[j])
                res2.append(output[j])
    
    actual_value = np.array(res1)
    predicted_value = np.array(res2)
    
    r2_res = 0
    rmse = 0
    rpd = 0
    mape = 0
    sample_length = actual_value.shape[1]
    for i in range(sample_length):
        r2_res += r2_score(actual_value[:, i], predicted_value[:, i])
        rmse += calculate_rmse(actual_value[:, i], predicted_value[:, i])
        rpd += calculate_rpd(actual_value[:, i], predicted_value[:, i])
        mape += calculate_mape(actual_value[:, i], predicted_value[:, i])
    
    r2_val = r2_res / sample_length
    rmse = rmse / sample_length
    rpd = rpd / sample_length
    mape = mape / sample_length
    return losses.avg, iteration, r2_val, rmse, rpd, mape

def validate(val_data_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    r2_val = 0
    res1 = []
    res2 = []
    for i, (data, label, plant_names) in enumerate(val_data_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        data = data.permute(0, 2, 1)
        output = model(data, None, None, None)
        loss = criterion(output, label)
        losses.update(loss.item())

        with torch.no_grad():
            output = output.data.cpu().numpy()
            label = label.cpu().numpy()
            for j in range(batch_size):
                res1.append(label[j])
                res2.append(output[j])
    
    actual_value = np.array(res1)
    predicted_value = np.array(res2)
    
    r2_res = 0
    rmse = 0
    rpd = 0
    mape = 0
    sample_length = actual_value.shape[1]
    for i in range(sample_length):
        r2_res += r2_score(actual_value[:, i], predicted_value[:, i])
        rmse += calculate_rmse(actual_value[:, i], predicted_value[:, i])
        rpd += calculate_rpd(actual_value[:, i], predicted_value[:, i])
        mape += calculate_mape(actual_value[:, i], predicted_value[:, i])
    
    r2_val = r2_res / sample_length
    rmse = rmse / sample_length
    rpd = rpd / sample_length
    mape = mape / sample_length
    return losses.avg, r2_val, rmse, rpd, mape

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error (RMSE) using sklearn"""
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_rpd(actual, predicted):
    """Calculate Relative Prediction Deviation (RPD)"""
    rmse = calculate_rmse(actual, predicted)
    sd = np.std(actual)
    return sd / rmse

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    ape = np.abs((actual - predicted) / actual)
    return np.mean(ape)

if __name__ == "__main__":
    main()
