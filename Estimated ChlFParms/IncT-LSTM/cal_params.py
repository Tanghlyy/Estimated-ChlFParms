#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :cal_params.py
@Description  :
@Time         :2024/05/27 21:27:06
@Author       :Tangh
@Version      :1.0
'''
import pandas as pd
import numpy as np
from scipy.integrate import trapz,simps
from scipy.interpolate import interp1d


class CalculateChlFParams():
    def __init__(self,data,columns=False):
        self.columns = columns
        data = data.values
        if(columns == True):
            self.data = data[:,1:459]
        else:
            self.data = data

    def get_basic_params(self,data):
        # 一行代表一条数据
        basic_dict = dict()
        bkg = data[:,0]
        basic_dict['Fo'] = data[:,3] - bkg # 对应40us处得到Fo的值
        basic_dict['F_300us'] = data[:,29] - bkg # 对应300us处的ChlF值
        basic_dict['Fj'] = data[:,71] - bkg # 对应2ms处得到Fj的值
        basic_dict['Fi'] = data[:,206] - bkg # 对应30ms处得到Fi的值
        basic_dict['Fm'] = np.max(data,axis=1) - bkg # 计算每一条OJIP序列中的最大荧光值
        basic_dict['Fv'] = basic_dict['Fm'] - basic_dict['Fo']
        return basic_dict


    def cal_der_params(self):
        der_dict = dict()
        # 获取时间间隔
        ojip_time = pd.read_csv('./data/time_scale_sampling.csv',header=None).values[:458,0].astype(float)
        basic_params = self.get_basic_params(self.data)
        #1 J点的相对可变叶绿素荧光
        der_dict['Vj'] = (basic_params['Fj'] - basic_params['Fo']) / basic_params['Fv']
        #2 I点的相对可变叶绿素荧光
        der_dict['Vi'] = (basic_params['Fi'] - basic_params['Fo']) / basic_params['Fv']
        #3 PSII最大光化学效率
        der_dict['Fv_Fm'] = basic_params['Fv'] / basic_params['Fm']
        #4 QA被还原的最大速率
        der_dict['Mo'] = 4 * (basic_params['F_300us'] - basic_params['Fo']) / basic_params['Fv']
        # 5 OJIP曲线与F=Fm之间的面积
        # der_dict['Area'] = self.cal_area(ojip_time,self.data)
        # #6 受体库容量，即Area标准化后的数据
        # der_dict['Sm'] = der_dict['Area'] / basic_params['Fv']
        ###############################################################
        #7 PSII最大光化学效率
        der_dict['Phi_Po'] = basic_params['Fv'] / basic_params['Fm']
        #8 捕获的激子将电子传递到电子传递链中超过QA的其他电子受体的概率
        der_dict['Psi_o'] = 1 - der_dict['Vj']
        #9 用于电子传递的量子产额
        der_dict['Phi_Eo'] = (1 - basic_params['Fo'] / basic_params['Fm']) * der_dict['Psi_o']
        ###############################################################
        #10 单位反应中心吸收的光能 
        der_dict['ABS_RC'] = der_dict['Mo'] * (1 / der_dict['Vj']) * (1 / der_dict['Phi_Po'])
        #11 单位反应中心捕获的光能
        der_dict['TRo_RC'] = der_dict['Mo'] * (1 / der_dict['Vj'])
        #12 单位反应中心用于电子传递的光能
        der_dict['ETo_RC'] = der_dict['Mo'] * (1 / der_dict['Vj']) * der_dict['Psi_o']
        #13 单位反应中心耗散掉的能量
        der_dict['DIo_RC'] = der_dict['ABS_RC'] - der_dict['TRo_RC']    
        ##############################################################
        #14 单位横截面积吸收的能量
        der_dict['ABS_CSo'] = basic_params['Fo']   
        #15 单位横截面积捕获的能量 
        der_dict['TRo_CSo'] = der_dict['Phi_Po'] * der_dict['ABS_CSo']
        #16 单位横截面积电子传递的量子产额
        der_dict['ETo_CSo'] = der_dict['Phi_Eo'] * der_dict['ABS_CSo']
        #17 单位横截面积耗散掉的能量
        der_dict['DIo_CSo'] = der_dict['ABS_CSo'] - der_dict['TRo_CSo']
        ###############################################################
        #18 单位面积上的反应中心的数量
        der_dict['RC_CSo'] = der_dict['Phi_Po'] * (der_dict['Vj'] / der_dict['Mo']) * der_dict['ABS_CSo']
        ###############################################################
        #19 电子传递到PSI末端的量子产额
        der_dict['delta_Ro'] = (1 - der_dict['Vi']) / (1 - der_dict['Vj'])
        #20 电子传递到PSI末端的量子效率
        der_dict['Phi_Ro'] =  (1 - basic_params['Fo'] / basic_params['Fm']) * (1 - der_dict['Vi'])
        ###############################################################
        #21 以吸收光能为基础的性能指数
        der_dict['PI_abs'] = (1 / der_dict['ABS_RC']) * (der_dict['Phi_Po'] / (1 - der_dict['Phi_Po'])) * (der_dict['Psi_o'] / (1 - der_dict['Psi_o']))
        #22 综合性能指数
        der_dict['PI_total'] = der_dict['PI_abs'] * der_dict['delta_Ro'] / (1 - der_dict['delta_Ro'])
        return der_dict
    
    def cal_area(self,time,raw_data):
        area = []
        print(raw_data.shape)
        for i in range(raw_data.shape[0]):
            index_of_first_max_value = np.argmax(raw_data[i])
            ojip_time = time[0:index_of_first_max_value+1]
            new_data = raw_data[i,0:index_of_first_max_value+1] - raw_data[i,0]       
            # print(index_of_first_max_value,len(ojip_time),len(new_data))       
            # 在新的时间上进行插值
            interp_func = interp1d(ojip_time, new_data,kind='cubic') 
            new_time = np.arange(ojip_time[0], ojip_time[-1], 10)
            interpolated_values = interp_func(new_time)
            area_simps = simps(interpolated_values, new_time) / 1000
            area.append(area_simps)
        return np.array(area)
            # print("Area using Simpson's rule:", area_simps)
    def read_excel_and_get_label_counts(self,df):
        # 获取第一列的标签
        labels = df.iloc[:, 0]
        # 统计每个类别的样本数量
        label_counts = labels.value_counts()
        return label_counts.to_dict()
# 

# raw_data = pd.read_pickle('./data/ChlFData_d.pkl')
# cal_params = CalculateChlFParams(data=raw_data,columns=True) #columns=Ture表示第一列有标签
# # # print(raw_data)
# # cal_params.cal_area(raw_data[:,0],raw_data[:,1])
# label_count = cal_params.read_excel_and_get_label_counts(raw_data)
# print(f'样本总数量为:{label_count}') # 15891 样本总数量为:{'水稻': 4896, '黑菜': 4623, '矮脚黄': 4037, 
#                                             #'冬季山茶花': 400, '桂花冬季': 379, '青椒': 356, 
#                                             # '大岛樱': 335, '冬青卫矛': 314, '山茶': 314, '桂花夏季': 237}

# print(cal_params.get_basic_params(cal_params.data))
# print(cal_params.cal_der_params())