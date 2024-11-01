#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :LoadDataset_Params_Raw.py
@Description  :
@Time         :2024/05/15 10:13:41
@Author       :Tangh
@Version      :1.0
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cal_params import CalculateChlFParams
import pandas as pd
import numpy as np
import torch
import random
from scipy.signal import stft

# 设置随机种子以确保结果的可重复性
random_seed = 42

class Fusion_OJIP_Dataset(Dataset):
    def __init__(self,file_path_l,file_path_d,train=True,val=False,train_ratio=0.8,val_ratio=0.1):
        self.train = train
        self.val = val
        self.train_samples = {}
        self.val_samples = {}
        self.test_samples = {}
        # 读取样本（光适应值）和标签（暗适应值）
        self.raw_data_l = pd.read_pickle(file_path_l)
        self.raw_data_d = pd.read_pickle(file_path_d)
        self.get_params_l = CalculateChlFParams(data=self.raw_data_l,columns=True)
        self.get_params_d = CalculateChlFParams(data=self.raw_data_d,columns=True)

        # 得到每种植物的种类label和数量count
        statistical_samples = self.get_params_d.read_excel_and_get_label_counts(self.raw_data_d)
        i = 1
        # 初始化随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        for label,total_count in statistical_samples.items():
            print(i,label,total_count)
            # 得到每一条数据的索引
            indices = self.raw_data_d[self.raw_data_d.iloc[:,0] == label].index.tolist()
            np.random.shuffle(indices) #打乱每种样本的索引,在每一类中进行打乱
            train_count = int(total_count * train_ratio)
            val_count = int(total_count * val_ratio)
            self.train_samples[label] = indices[:train_count]
            self.val_samples[label] = indices[train_count:train_count+val_count]
            self.test_samples[label] = indices[train_count+val_count:]
            i += 1
        self.samples = []
        if self.train:
            for label,indices in self.train_samples.items():
                self.samples.extend(indices)
        elif self.val:
            for label, indices in self.val_samples.items():
                self.samples.extend(indices)
        else:
            for label,indices in self.test_samples.items():
                self.samples.extend(indices)
        print(f"总样本数：{len(self.samples)}")


        der_params_l = self.get_params_l.cal_der_params() # 得到导出参数（dict）
        der_params_d = self.get_params_d.cal_der_params()
        self.label = list(der_params_d.values()) # 将导出参数添加到列表中
        self.label_array = np.column_stack(self.label) # 将导出参数转换为array
        # 获取样本类别：
        self.sample_names = self.raw_data_l.values[:,0]
        # 拼接参数(20个)与原始数据，其中参数在前
        self.new_raw_d = np.hstack((self.label_array, self.raw_data_d.values[:,1:459])).astype(np.float32)
        self.new_raw_l = self.raw_data_l.values[:,1:459].astype(np.float32)
        print(self.new_raw_d.shape,self.new_raw_l.shape)
        # print(self.new_raw_d[0,:])
        # print(self.new_raw_l[0,:])
        
        self.new_min_l = np.min(self.new_raw_l,axis=0)
        self.new_max_l = np.max(self.new_raw_l,axis=0)
        self.new_mean_l = np.mean(self.new_raw_l,axis=0)
        self.new_std_l = np.std(self.new_raw_l,axis=0)

        self.new_min_d = np.min(self.new_raw_d,axis=0)
        self.new_max_d = np.max(self.new_raw_d,axis=0)
        self.new_mean_d = np.mean(self.new_raw_d,axis=0)
        self.new_std_d = np.std(self.new_raw_d,axis=0)
       
    def __getitem__(self, index):
        data_index = self.samples[index] #获取数据打乱后的索引
        plant_name = self.sample_names[data_index]
        label = self.new_raw_d[data_index].astype(np.float32)
        input = self.new_raw_l[data_index].astype(np.float32)
        label = (label - self.new_mean_d) / self.new_std_d
        # 对输入进行数据加强（1、最小-最大归一化 2、Z-Score归一化）
        out = np.empty((1,458)).astype(np.float32)
        out[0, :] = (input - self.new_mean_l) / self.new_std_l 
        out = torch.as_tensor(out)
        return out, label
    
    def __len__(self):
        return len(self.samples)
    
    
# # 创建训练集和测试集的数据集对象
# train_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=True,val=False,train_ratio=0.6,val_ratio=0.2)
# val_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=True,train_ratio=0.6,val_ratio=0.2)
# test_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=False,train_ratio=0.6,val_ratio=0.2)

# # 创建DataLoader
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # # 测试输出
# # for sample in train_loader:
# #     print("训练集样本：", sample)
# #     break  # 只打印一个batch的数据

# # for sample in val_loader:
# #     print("测试集样本：", sample)
# #     break  # 只打印一个batch的数据

# # for sample in test_loader:
# #     print("测试集样本：", sample)
# #     break  # 只打印一个batch的数据

# for i, (data, label,plant_names) in enumerate(test_loader):
#     print(i,plant_names)
