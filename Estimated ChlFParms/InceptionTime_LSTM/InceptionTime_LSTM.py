import torch
import torch.nn as nn
import torch.nn.functional as F
from Parallel_Inception_Layer import Parallel_Inception_Layer
from A_few_samepadding_layers import ShortcutLayer, SampaddingConv1D, SampaddingMaxPool1D
from torch.nn.parameter import Parameter

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.TA = nn.Sequential(
        #     nn.ConstantPad1d((int((5 - 1) / 2), int(5 / 2)), 0),
        #     nn.Conv1d(1, 1, 5, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        b, c, _ = x.size()
        # TA = torch.mean(x, dim=1, keepdim=True)
        # TA = self.TA(TA)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        # x = x * TA.expand_as(x)
        return x


class Inception_module(nn.Module):
    def __init__(self, input_channel_size, nb_filters, bottleneck_size, kernel_sizes, stride=1, activation='linear'):
        super(Inception_module, self).__init__()
        self.input_channle_size = input_channel_size
        self.nb_filters = nb_filters
        self.bottleneck_size = bottleneck_size
        self.kernel_sizes = kernel_sizes - 1
        self.stride = stride
        self.activation = activation

        self.n_incepiton_scale = 25
        self.kernel_size_s = [2 * i + 3 for i in range(self.n_incepiton_scale)]
        # print(self.kernel_size_s)
        # ori 20 2*i+3

        if self.input_channle_size > 1 and self.bottleneck_size != None:
            self.bottleneck_layer = SampaddingConv1D(self.input_channle_size, self.bottleneck_size, kernel_size=1,
                                                     use_bias=False)
            self.layer_parameter_list = [(self.bottleneck_size, self.nb_filters, kernel_size) for kernel_size in
                                         self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list, use_bias=False,
                                                                     use_batch_Norm=False, use_relu=False)
        else:
            self.layer_parameter_list = [(self.input_channle_size, self.nb_filters, kernel_size) for kernel_size in
                                         self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list, use_bias=False,
                                                                     use_batch_Norm=False, use_relu=False)

        self.maxpooling_layer = SampaddingMaxPool1D(3, self.stride)
        self.conv_6_layer = SampaddingConv1D(self.input_channle_size, self.nb_filters, kernel_size=1, use_bias=False)

        self.output_channel_numebr = self.nb_filters * (self.n_incepiton_scale + 1)
        self.bn_layer = nn.BatchNorm1d(num_features=self.output_channel_numebr)

    def forward(self, X):
        if X.shape[-2] > 1:
            input_inception = self.bottleneck_layer(X)
        else:
            input_inception = X
        concatenateed_conv_list_result = self.parallel_inception_layer(input_inception)
        conv_6 = self.conv_6_layer(self.maxpooling_layer(X))

        concatenateed_conv_list_result_2 = torch.cat((concatenateed_conv_list_result, conv_6), 1)
        result = F.relu(self.bn_layer(concatenateed_conv_list_result_2))
        # result = F.relu(concatenateed_conv_list_result_2)
        return result


class InceptionNet(nn.Module):
    def __init__(self,
                 input_channle_size,
                 nb_classes,
                 is_noise,
                 batch_size,
                 verbose=False,
                 build=True,
                 nb_filters=36,
                 use_residual=True,
                 use_bottleneck=True,
                 depth=9,
                 kernel_size=50,
                 device='cuda'):
        super(InceptionNet, self).__init__()

        self.input_channle_size = input_channle_size
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.is_noise = is_noise
        self.batch_size = batch_size
        self.device = device
        if use_bottleneck:
            self.bottleneck_size = 32
        else:
            self.bottleneck_size = None

        self.res_layer_list = nn.ModuleList()
        self.layer_list = nn.ModuleList()
        self.SE_list = nn.ModuleList()
        self.out_put_channle_number_list = []
        # self.para_list = []

        for d in range(self.depth):
            if d == 0:
                input_channle_size_for_this_layer = self.input_channle_size
            else:
                input_channle_size_for_this_layer = self.out_put_channle_number_list[-1]
            inceptiontime_layer = Inception_module(input_channle_size_for_this_layer,
                                                   self.nb_filters,
                                                   self.bottleneck_size,
                                                   self.kernel_size,
                                                   stride=1,
                                                   activation='linear')
            self.layer_list.append(inceptiontime_layer)
            SE_layer = SELayer(inceptiontime_layer.output_channel_numebr)
            self.SE_list.append(SE_layer)
            self.out_put_channle_number_list.append(inceptiontime_layer.output_channel_numebr)

            if self.use_residual and d % 3 == 2:
                if d == 2:
                    shortcutlayer = ShortcutLayer(self.input_channle_size, self.out_put_channle_number_list[-1],
                                                  kernel_size=1, use_bias=False)
                else:
                    shortcutlayer = ShortcutLayer(self.out_put_channle_number_list[-4],
                                                  self.out_put_channle_number_list[-1], kernel_size=1, use_bias=False)
                self.res_layer_list.append(shortcutlayer)
                # para = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                # para = para.to(device=device)
                # para.data.fill_(1)
                # self.para_list.append(para)

        self.averagepool = nn.AdaptiveAvgPool1d(1)
        # self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.lstm = nn.LSTM(input_size=self.input_channle_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.blstm = nn.LSTM(self.input_channle_size, 128, batch_first=False, bidirectional=True)
        # self.GRU = nn.GRU(input_size=self.input_channle_size,
        #                   hidden_size=128,
        #                   num_layers=2,
        #                   batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.Linear(128*2, 256)
        # self.BN=nn.BatchNorm1d(self.out_put_channle_number_list[-1] + 128)
        self.hidden = nn.Linear(self.out_put_channle_number_list[-1] + 128, self.nb_classes)
        # self.hidden = nn.Linear(128, self.nb_classes)
        # self.Linear1 = nn.Linear(23040, 1000)
        # self.Linear2 = nn.Linear(1000, 50)
        # self.Linear3 = nn.Linear(50,self.nb_classes)
        # self.softmax = nn.Softmax(dim=1)
        # self.reduce = nn.Sequential(nn.Linear(457, 200))
        # self.kernel = Poly(457)

    def init_hidden(self):
        h0 = torch.zeros(2, self.batch_size, 128).to(self.device)
        c0 = torch.zeros(2, self.batch_size, 128).to(self.device)
        return h0, c0

    def att_dot_seq_len(self, x):
        # b, s, input_size / b, s, hidden_size
        x = self.attention(x)  # bsh--->bst
        e = torch.bmm(x, x.permute(0, 2, 1))  # bst*bts=bss
        attention = F.softmax(e, dim=-1)  # b s s
        out = torch.bmm(attention, x)  # bss * bst ---> bst
        out = F.relu(out)

        return out

    def forward(self, X):
        h0, c0 = self.init_hidden()
        x = X.transpose(2, 1)  # [B, T, F]
        x1, (ht, ct) = self.lstm(x, (h0, c0))
        x1 = x1[:, -1, :]
        # x2, ht = self.GRU(x, h0)
        # x2 = x2[:, -1, :]
        # x2, _ = self.blstm(x)
        # x2 = self.att_dot_seq_len(x2)
        # x2 = x2[:, -1, :]

        res_layer_index = 0
        input_res = X
        for d in range(self.depth):
            X = self.layer_list[d](X)
            X = self.SE_list[d](X)
            if self.use_residual and d % 3 == 2:
                shot_cut = self.res_layer_list[res_layer_index](input_res)
                res_layer_index = res_layer_index + 1
                X = torch.add(shot_cut, X)
                input_res = X
        XA = self.averagepool(X)
        XA = XA.squeeze_(-1)
        X = torch.cat((x1, XA), dim=1)
        X = self.dropout(X)

        # XM = self.maxpool(X)
        # XM = XM.squeeze_(-1)

        # X = self.BN(X)

        X = self.hidden(X)
        return X
    

