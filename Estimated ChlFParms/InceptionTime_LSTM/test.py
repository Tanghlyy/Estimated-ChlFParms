#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :test.py
@Description  :
@Time         :2024/06/21 14:27:53
@Author       :Tangh
@Version      :1.0
'''

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from InceptionTime_LSTM import InceptionNet  # 确保你的 InceptionNet 类正确导入
from LoadDataset_Params_Raw import Fusion_OJIP_Dataset
from sklearn.metrics import r2_score
import numpy as np

# 参数设置
batch_size = 16
features = 478
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model_path = './models-1channel/ChlF_Model/best.pkl'  # 使用保存的最佳模型
model = InceptionNet(1, features, False, batch_size)
model = nn.DataParallel(model)
model = model.to(device=device, dtype=torch.float)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# 准备测试数据
test_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=False,train_ratio=0.6,val_ratio=0.2)
test_data_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)


def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iter']

    return epoch, iteration


print(load_model(model_path,model))



# 进行测试
model.eval()
res1 = []
res2 = []
res3 = []
name_list = []

with torch.no_grad():
    for i, (data, label,plant_name) in enumerate(test_data_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        # plant_name = plant_name.to(device=device)
        # 预测
        output = model(data)      
        # 将预测结果和真实结果添加到列表中
        output = output.data.cpu().numpy()
        label = label.cpu().numpy()
        input_l = data.cpu().numpy()[:,0,:]
        for i in range(batch_size):
            data_l = input_l[i]
            out_data = output[i]  # 预测值
            test_label = label[i]  # 真实值
            name = plant_name[i] # 获取每类样本的名称
            out_data = out_data * test_dataset.new_std_d[:features] + test_dataset.new_mean_d[:features]
            test_label = test_label * test_dataset.new_std_d[:features] + test_dataset.new_mean_d[:features]
            data_l = data_l * test_dataset.new_std_l + test_dataset.new_mean_l
            res1.append(test_label)
            res2.append(out_data)
            res3.append(data_l)
            name_list.append(name)

# 转换为numpy数组
res1 = np.array(res1)
res2 = np.array(res2)
res3 = np.array(res3)
name_list = np.array(name_list)
# 计算 R² 值
r2_res = 0
for i in range(res1.shape[1]):
    r2_res += r2_score(res1[:,i], res2[:,i])
r2_test = r2_res / res1.shape[1]


data_list = []
for i in range(len(res1)):
    data_list.append(np.concatenate((res1[i, :], res2[i, :], res3[i, :]), axis=0))
df = pd.DataFrame(data_list)
df.insert(0,'Plant Name',name_list)
# Save to Excel
df.to_excel("test_output.xlsx", index=False)

print(f'Test R²: {r2_test:.4f}')