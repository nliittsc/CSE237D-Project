# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:08:48 2021

@author: 18315
"""


import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
import os


#%%

def bp_to_category(y):
    """
    Parameters
    ----------
    y : (float, array-like)
        1d Vector with 2 elements: y[0] = systolic BP and y[1] = diastolic BP

    Returns
    -------
    int
        Integer categorical feature denoting a persons blood pressure category.
        Feature is ordinal: higher numbers mean more "severe" blood pressure

    """
    if y[0] < 120 and y[1] < 80:
        return 0
    elif (120 <= y[0] < 130) and (y[1] < 80):
        return 1
    elif (130 <= y[0] < 140) or (80 <= y[1] < 90):
        return 2
    elif (140 <= y[0] <= 160) or (90 <= y[1] <= 120):
        return 3
    else:
        return 4

class DataConstructor:
    def __init__(self, directory):
        self.directory = directory
        self.test_dataset = None
        self.pretrain_dataset = None
        
    def load_pretrain_data(self):
        path = r"C:\Users\18315\Documents\Projects\SmartBP\BPDataAuscOsc\BPDataAuscOsc.mat"
        data_set = loadmat(path)

        d1 = data_set['AuscWave'][0]
        d2 = data_set['CuffPressure'][0]
        d3 = data_set['DBP'][0]
        d4 = data_set['SBP'][0]
        d5 = data_set['oscNNCell'][0]
        
        data = dict()
        for i in range(len(d1)):
            data[i] = dict()
            #data[i]['AuscWave'] = np.flip(d1[i].reshape(-1))
            data[i]['ap'] = np.flip(d2[i].reshape(-1))
            sbp = d4[i][0][0]
            dbp = d3[i][0][0]
            data[i]['bp'] = [sbp, dbp] 
            data[i]['bp_cat'] = bp_to_category([sbp, dbp])
            data[i]['ppg'] = np.flip(d5[i].reshape(-1))

        self.pretrain_dataset = data
        return self.pretrain_dataset

        
    def load_test_data(self, directory=None):
        if directory is None:
            directory = self.directory
            
        data = []
        for filename in os.listdir(directory):
            data_dict = dict()  # To store the signal
            
            # load in each signal
            path = os.path.join(directory, filename)
            d = pd.read_csv(path)
            
            data_dict['time'] = d.time.values
            data_dict['ppg'] = d.ppg.values
            data_dict['ap'] = d.force.values
            sbp = d.sys_bp.values[0]  # Systolic BP
            dbp = d.dia_bp.values[0]  # Diastolic BP
            data_dict['bp'] = [sbp, dbp]
            data_dict['bp_cat'] = bp_to_category([sbp, dbp])
            
            data.append(data_dict)
        self.test_dataset = pd.DataFrame(data)
        return self.test_dataset
        
