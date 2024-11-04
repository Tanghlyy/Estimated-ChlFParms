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

# Set random seed to ensure reproducibility
random_seed = 42

class Fusion_OJIP_Dataset(Dataset):
    def __init__(self,file_path_l,file_path_d,train=True,val=False,train_ratio=0.8,val_ratio=0.1):
        self.train = train
        self.val = val
        self.train_samples = {}
        self.val_samples = {}
        self.test_samples = {}
        # Read samples (light-adapted values) and labels (dark-adapted values)
        self.raw_data_l = pd.read_pickle(file_path_l)
        self.raw_data_d = pd.read_pickle(file_path_d)
        self.get_params_l = CalculateChlFParams(data=self.raw_data_l,columns=True)
        self.get_params_d = CalculateChlFParams(data=self.raw_data_d,columns=True)

        # Get the label and count of each plant species
        statistical_samples = self.get_params_d.read_excel_and_get_label_counts(self.raw_data_d)
        i = 1
        # Initialize random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        for label,total_count in statistical_samples.items():
            print(i,label,total_count)
            # Get the index of each data point
            indices = self.raw_data_d[self.raw_data_d.iloc[:,0] == label].index.tolist()
            np.random.shuffle(indices) # Shuffle the indices of each sample within each category
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
        print(f"Total number of samples: {len(self.samples)}")


        der_params_l = self.get_params_l.cal_der_params() # Get derived parameters (dict)
        der_params_d = self.get_params_d.cal_der_params()
        self.label = list(der_params_d.values()) # Add derived parameters to the list
        self.label_array = np.column_stack(self.label) # Convert derived parameters to array
        # Get sample categories:
        self.sample_names = self.raw_data_l.values[:,0]
        # Concatenate parameters (20) with raw data, with parameters in front
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
        data_index = self.samples[index] # Get the shuffled index of the data
        plant_name = self.sample_names[data_index]
        label = self.new_raw_d[data_index].astype(np.float32)
        input = self.new_raw_l[data_index].astype(np.float32)
        label = (label - self.new_mean_d) / self.new_std_d
        # Data augmentation for input (1. Min-Max normalization 2. Z-Score normalization)
        out = np.empty((1,458)).astype(np.float32)
        # out[0, :] = (input - self.new_min_l) / (self.new_max_l - self.new_min_l) * 2 - 1
        out[0, :] = (input - self.new_mean_l) / self.new_std_l 
        # out[2, :] = out[0, :][::-1]
        # out[3, :] = out[1, :][::-1]
        # out[4:, :] = -out[:4, :]
        out = torch.as_tensor(out)
        return out, label, plant_name # label[0:20]
    
    def __len__(self):
        return len(self.samples)
    
    
# # Create dataset objects for training and testing sets
# train_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=True,val=False,train_ratio=0.6,val_ratio=0.2)
# val_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=True,train_ratio=0.6,val_ratio=0.2)
# test_dataset = Fusion_OJIP_Dataset('./data/ChlFData_l.pkl', './data/ChlFData_d.pkl', train=False,val=False,train_ratio=0.6,val_ratio=0.2)

# # Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # # Test output
# # for sample in train_loader:
# #     print("Training set sample:", sample)
# #     break  # Only print one batch of data

# # for sample in val_loader:
# #     print("Validation set sample:", sample)
# #     break  # Only print one batch of data

# # for sample in test_loader:
# #     print("Test set sample:", sample)
# #     break  # Only print one batch of data

# for i, (data, label,plant_names) in enumerate(test_loader):
#     print(i,plant_names)
