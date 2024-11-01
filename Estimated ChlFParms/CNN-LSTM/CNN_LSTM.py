#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :CNN-LSTM.py
@Description  :
@Time         :2024/07/17 14:16:56
@Author       :Tangh
@Version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, input_size=458, num_classes=478):
        super(CNNLSTM, self).__init__()

        # Encoder 1
        self.ec1_layer1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=20, stride=2)
        self.ec1_layer2 = nn.Conv1d(in_channels=50, out_channels=30, kernel_size=10, stride=2)
        self.ec1_pool = nn.MaxPool1d(kernel_size=2)

        # Encoder 2
        self.ec2_layer1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=6, stride=1)
        self.ec2_layer2 = nn.Conv1d(in_channels=50, out_channels=40, kernel_size=6, stride=1)
        self.ec2_pool1 = nn.MaxPool1d(kernel_size=2)
        self.ec2_layer3 = nn.Conv1d(in_channels=40, out_channels=30, kernel_size=6, stride=1)
        self.ec2_layer4 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=6, stride=2)
        self.ec2_pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(input_size=30, hidden_size=60, num_layers=2, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(60, 60)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(60, num_classes)
    
    def forward(self, x):
        # Encoder 1
        x1 = F.tanh(self.ec1_layer1(x))
        x1 = F.tanh(self.ec1_layer2(x1))
        x1 = self.ec1_pool(x1)

        # Encoder 2
        x2 = F.tanh(self.ec2_layer1(x))
        x2 = F.tanh(self.ec2_layer2(x2))
        x2 = self.ec2_pool1(x2)
        x2 = F.tanh(self.ec2_layer3(x2))
        x2 = F.tanh(self.ec2_layer4(x2))
        x2 = self.ec2_pool2(x2)

        # Combine encoders
        x = x1 * x2

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, input_size)

        # LSTM
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])  # Use the last output of the LSTM
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


