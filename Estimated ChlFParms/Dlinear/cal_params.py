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
        # Each row represents one data point
        basic_dict = dict()
        bkg = data[:,0]
        basic_dict['Fo'] = data[:,3] - bkg # Get Fo value at 40us
        basic_dict['F_300us'] = data[:,29] - bkg # ChlF value at 300us
        basic_dict['Fj'] = data[:,71] - bkg # Get Fj value at 2ms
        basic_dict['Fi'] = data[:,206] - bkg # Get Fi value at 30ms
        basic_dict['Fm'] = np.max(data,axis=1) - bkg # Calculate the maximum fluorescence value in each OJIP sequence
        basic_dict['Fv'] = basic_dict['Fm'] - basic_dict['Fo']
        return basic_dict


    def cal_der_params(self):
        der_dict = dict()
        # Get time intervals
        ojip_time = pd.read_csv('./data/time_scale_sampling.csv',header=None).values[:458,0].astype(float)
        basic_params = self.get_basic_params(self.data)
        #1 Relative variable chlorophyll fluorescence at J point
        der_dict['Vj'] = (basic_params['Fj'] - basic_params['Fo']) / basic_params['Fv']
        #2 Relative variable chlorophyll fluorescence at I point
        der_dict['Vi'] = (basic_params['Fi'] - basic_params['Fo']) / basic_params['Fv']
        #3 Maximum photochemical efficiency of PSII
        der_dict['Fv_Fm'] = basic_params['Fv'] / basic_params['Fm']
        #4 Maximum rate of QA reduction
        der_dict['Mo'] = 4 * (basic_params['F_300us'] - basic_params['Fo']) / basic_params['Fv']
        # 5 Area between OJIP curve and F=Fm
        # der_dict['Area'] = self.cal_area(ojip_time,self.data)
        # #6 Receptor pool capacity, i.e., Area normalized data
        # der_dict['Sm'] = der_dict['Area'] / basic_params['Fv']
        ###############################################################
        #7 Maximum photochemical efficiency of PSII
        der_dict['Phi_Po'] = basic_params['Fv'] / basic_params['Fm']
        #8 Probability of excitons captured transferring electrons to other electron acceptors beyond QA in the electron transport chain
        der_dict['Psi_o'] = 1 - der_dict['Vj']
        #9 Quantum yield for electron transport
        der_dict['Phi_Eo'] = (1 - basic_params['Fo'] / basic_params['Fm']) * der_dict['Psi_o']
        ###############################################################
        #10 Light energy absorbed per reaction center
        der_dict['ABS_RC'] = der_dict['Mo'] * (1 / der_dict['Vj']) * (1 / der_dict['Phi_Po'])
        #11 Light energy captured per reaction center
        der_dict['TRo_RC'] = der_dict['Mo'] * (1 / der_dict['Vj'])
        #12 Light energy used for electron transport per reaction center
        der_dict['ETo_RC'] = der_dict['Mo'] * (1 / der_dict['Vj']) * der_dict['Psi_o']
        #13 Energy dissipated per reaction center
        der_dict['DIo_RC'] = der_dict['ABS_RC'] - der_dict['TRo_RC']    
        ##############################################################
        #14 Energy absorbed per cross-sectional area
        der_dict['ABS_CSo'] = basic_params['Fo']   
        #15 Energy captured per cross-sectional area
        der_dict['TRo_CSo'] = der_dict['Phi_Po'] * der_dict['ABS_CSo']
        #16 Quantum yield for electron transport per cross-sectional area
        der_dict['ETo_CSo'] = der_dict['Phi_Eo'] * der_dict['ABS_CSo']
        #17 Energy dissipated per cross-sectional area
        der_dict['DIo_CSo'] = der_dict['ABS_CSo'] - der_dict['TRo_CSo']
        ###############################################################
        #18 Number of reaction centers per area
        der_dict['RC_CSo'] = der_dict['Phi_Po'] * (der_dict['Vj'] / der_dict['Mo']) * der_dict['ABS_CSo']
        ###############################################################
        #19 Quantum yield of electron transport to the end of PSI
        der_dict['delta_Ro'] = (1 - der_dict['Vi']) / (1 - der_dict['Vj'])
        #20 Quantum efficiency of electron transport to the end of PSI
        der_dict['Phi_Ro'] =  (1 - basic_params['Fo'] / basic_params['Fm']) * (1 - der_dict['Vi'])
        ###############################################################
        #21 Performance index based on absorbed light energy
        der_dict['PI_abs'] = (1 / der_dict['ABS_RC']) * (der_dict['Phi_Po'] / (1 - der_dict['Phi_Po'])) * (der_dict['Psi_o'] / (1 - der_dict['Psi_o']))
        #22 Comprehensive performance index
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
            # Interpolate on new time
            interp_func = interp1d(ojip_time, new_data,kind='cubic') 
            new_time = np.arange(ojip_time[0], ojip_time[-1], 10)
            interpolated_values = interp_func(new_time)
            area_simps = simps(interpolated_values, new_time) / 1000
            area.append(area_simps)
        return np.array(area)
            # print("Area using Simpson's rule:", area_simps)
    def read_excel_and_get_label_counts(self,df):
        # Get labels from the first column
        labels = df.iloc[:, 0]
        # Get labels from the first column
        label_counts = labels.value_counts()
        return label_counts.to_dict()
# 
