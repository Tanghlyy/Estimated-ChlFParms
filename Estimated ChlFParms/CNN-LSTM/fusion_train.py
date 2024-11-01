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
# import torchinfo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# from model import base

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #指定使用GPU的ID
import time
import argparse
from LoadDataset_Params_Raw import Fusion_OJIP_Dataset

from CNN_LSTM import CNNLSTM
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss
import numpy as np
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='Image Classification:') #创建参数解析器对象
parser.add_argument('--model_path', type=str, default='ChlF_Model',
                    help="Set model storage path") #添加参数

args = parser.parse_args() #解析命令行参数
file_path_l = './data/ChlFData_l.pkl'
file_path_d = './data/ChlFData_d.pkl'
train_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=True,val=False,train_ratio=0.6,val_ratio=0.2)
val_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=True,train_ratio=0.6,val_ratio=0.2)
batch_size = 16 
# model input  feature size is 478(all) or 20(params)
features = 478
def main():
    cudnn.benchmark = True #优化网络运行性能
    model = CNNLSTM(input_size=458,num_classes=features)
    # torchinfo.summary(model)
    # model = MLSTMfcn(num_classes=4, max_seq_len=457, num_features=4)
    # model = WideResNet(8, 4, 0, 4, 4)
    # model = LSTM_FCN(2, 1, 3)
    # multi-GPU setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)

    model = model.to(device=device, dtype=torch.float)

    start_epoch = 0
    end_epoch = 201
    init_lr = 1e-3
    # init_lr = 0.005
    iteration = 0
    acc_ten = np.empty(10)
    #
    criterion = nn.MSELoss()
    # criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False,
                                  threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=1e-10, eps=1e-8)
    model_path = './models-622/' + args.model_path
    max_acc = 0
    acc_index = 0

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    # resume_file = 'models/models/base_3_16channels/ssfsr_9layers_epoch1200.pkl'
    resume_file = ''
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch + 1, end_epoch):
        torch.cuda.empty_cache()
        train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)
        val_data_loader = DataLoader(dataset=val_dataset, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=True,drop_last=True)
  
        start_time = time.time()
        train_loss, iteration, train_r2, train_rmse, train_rpd, train_mape= train(train_data_loader, model, criterion, optimizer, iteration, device)
        val_loss, val_r2, val_rmse, val_rpd, val_mape = validate(val_data_loader, model, criterion, device)

        if max_acc < val_r2:
            max_acc = val_r2
            save_checkpoint(model_path, epoch, iteration, model, optimizer, name='best')
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        scheduler.step(val_loss)

        # Save model
        if epoch % 50 == 0:
            save_checkpoint(model_path, epoch, iteration, model, optimizer,name=str(epoch)+'epoch')

        end_time = time.time()
        epoch_time = end_time - start_time
        if acc_index < 9:
            acc_ten[acc_index] = val_r2
            print("Epoch [%d]|Iter[%d]|Time:%.4f|learning rate : %.9f|Train Loss: %.4f|Train R²:%.2f%%|Train RMSE:%.4f|Train RPD:%.4f|Train MAPE:%.4f|"
                  "Val Loss:%.9f|Val R²:%.2f%%|Val RMSE:%.4f|Val RPD:%.4f|Val MAPE:%.4f|" % (
                      epoch, iteration, epoch_time, lr, train_loss, train_r2 * 100,train_rmse,train_rpd,train_mape,
                      val_loss, val_r2 * 100,val_rmse,val_rpd,val_mape))
            record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss,train_r2,val_r2,train_rmse,val_rmse,train_rpd,val_rpd,train_mape,val_mape)
            logger.info(
                "Epoch [%d], Iter[%d], Time:%.4f, learning rate : %.9f, Train Loss: %.4f  Train R²:%.2f%%, "
                "Val Loss:%.9f  Val R²:%.2f%%   Near 10 Accuracy:None" % (
                    epoch, iteration, epoch_time, lr, train_loss, train_r2 * 100, val_loss, val_r2 * 100))
        else:
            acc_ten[acc_index % 10] = val_r2
            print("Epoch [%d]|Iter[%d]|Time:%.4f|learning rate : %.9f|Train Loss: %.4f|Train R²:%.2f%%|Train RMSE:%.4f|Train RPD:%.4f|Train MAPE:%.4f|"
                  "Val Loss:%.9f|Val R²:%.2f%%|Val RMSE:%.4f|Val RPD:%.4f|Val MAPE:%.4f|" % (
                      epoch, iteration, epoch_time, lr, train_loss, train_r2 * 100,train_rmse,train_rpd,train_mape,
                      val_loss, val_r2 * 100,val_rmse,val_rpd,val_mape))
            record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss,train_r2,val_r2,train_rmse,val_rmse,train_rpd,val_rpd,train_mape,val_mape)
            logger.info(
                "Epoch [%d], Iter[%d], Time:%.4f, learning rate : %.9f, Train Loss: %.4f  Train R²:%.2f%%, "
                "Val Loss:%.4f  Val R²:%.2f%%   Near 10 Accuracy:%.2f%%" % (
                    epoch, iteration, epoch_time, lr, train_loss, train_r2 * 100, val_loss, val_r2 * 100,
                    acc_ten.mean() * 100))
        acc_index += 1


def train(train_data_loader, model, criterion, optimizer, iteration, device):
    losses = AverageMeter()
    model.train()
    r2_train = 0
    res1 = []
    res2 = []
    for i, (data, label,plant_names) in enumerate(train_data_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        iteration = iteration + 1
        # lam = np.random.beta(1,1)
        # index =torch.randperm(data.size(0)).cuda()
        # mixed_x = lam*data+(1-lam)*data[index,:]
        # Forward + Backward + Optimize
        # output = model(mixed_x)
        # loss = lam*criterion(output, label)+(1-lam)*criterion(output,label)
        output = model(data)
        loss = criterion(output, label) #+ 0.1 * OJIP_params_loss(output, label) + 0.1 * tv_loss(output)
        optimizer.zero_grad()
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        #  record loss
        losses.update(loss.item())
        with torch.no_grad():
            label = label.cpu().numpy()
            output = output.data.cpu().numpy()
            for i in range(batch_size):
                out_data = output[i]
                train_label = label[i]
                res1.append(train_label)
                res2.append(out_data)
    actual_value = np.array(res1)
    predicted_value = np.array(res2)
    # print(res1.shape,res2.shape)
    r2_res = 0
    rmse = 0
    rpd = 0
    mape = 0
    sample_length = actual_value.shape[1]
    for i in range(sample_length):
        r2_res += r2_score(actual_value[:,i], predicted_value[:,i])
        rmse += calculate_rmse(actual_value[:,i],predicted_value[:,i])
        rpd += calculate_rpd(actual_value[:,i],predicted_value[:,i])
        mape += calculate_mape(actual_value[:,i],predicted_value[:,i])
    r2_val = r2_res / sample_length
    rmse = rmse / sample_length
    rpd = rpd / sample_length
    mape = mape / sample_length
    return losses.avg, iteration,r2_val,rmse,rpd,mape

def validate(val_data_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    r2_val = 0
    res1 = []
    res2 = []
    for i, (data, label,plant_names) in enumerate(val_data_loader):
        data = data.to(device=device)
        label = label.to(device=device)

        # compute output
        output = model(data)
        loss = criterion(output, label) #+ 0.1 * OJIP_params_loss(output, label) + 0.1 * tv_loss(output)

        #  record loss
        losses.update(loss.item())
        # result = torch.argmax(output.data, 1) == torch.argmax(label, 1)
        # # result = torch.argmax(output.data, 1) == label
        # result = result.cpu().numpy()
        # correct_num += np.count_nonzero(result)
        # data_num += len(result)
        output = output.data.cpu().numpy()
        label = label.cpu().numpy()
        for i in range(batch_size):
            out_data = output[i] #预测值
            val_label = label[i] #真实值
            # out_data = out_data * val_dataset.new_std_d[:features] + val_dataset.new_mean_d[:features]
            # val_label = val_label * val_dataset.new_std_d[:features] + val_dataset.new_mean_d[:features]
            res1.append(val_label)
            res2.append(out_data)
    actual_value = np.array(res1)
    predicted_value = np.array(res2)
    # print(res1.shape,res2.shape)
    r2_res = 0
    rmse = 0
    rpd = 0
    mape = 0
    sample_length = actual_value.shape[1]
    for i in range(sample_length):
        r2_res += r2_score(actual_value[:,i], predicted_value[:,i])
        rmse += calculate_rmse(actual_value[:,i],predicted_value[:,i])
        rpd += calculate_rpd(actual_value[:,i],predicted_value[:,i])
        mape += calculate_mape(actual_value[:,i],predicted_value[:,i])
    r2_val = r2_res / sample_length
    rmse = rmse / sample_length
    rpd = rpd / sample_length
    mape = mape / sample_length
    return losses.avg, r2_val,rmse,rpd,mape


def tv_loss(output):
    # torch.size(4,457)
    dx = torch.abs(output[:,:-1] - output[:,1:])
    return torch.sum(dx)


def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error (RMSE) using sklearn"""
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_rpd(actual, predicted):
    """Calculate Relative Prediction Deviation (RPD)"""
    rmse = calculate_rmse(actual, predicted)
    sd = np.std(actual) 
    return sd / rmse

def calculate_mape(actual, predicted):
    """Calculate Median Percentage Absolute Error (MPAE)"""
    ape = np.abs((actual - predicted) / actual) 
    return np.mean(ape)

if __name__ == "__main__":
    main()
