# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:50:14 2021

@author: 18315
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.matlab import mio
from scipy import signal

path = r"C:\Users\18315\Documents\Projects\SmartBP\Data File\PPG-BP dataset.xlsx"
data = pd.read_excel(path, skiprows=1)

path2_1 = r"C:\Users\18315\Documents\Projects\SmartBP\Data File\0_subject\6_1.txt"
path2_2 = r"C:\Users\18315\Documents\Projects\SmartBP\Data File\0_subject\6_2.txt"
path2_3 = r"C:\Users\18315\Documents\Projects\SmartBP\Data File\0_subject\6_3.txt"
d2_1 = pd.read_csv(path2_1)
d2_1 = [float(x) for x in d2_1.columns[0].split()]
d2_2 = pd.read_csv(path2_2)
d2_2 = [float(x) for x in d2_2.columns[0].split()]
d2_3 = pd.read_csv(path2_3)
d2_3 = [float(x) for x in d2_3.columns[0].split()]

bpm = data[data.subject_ID == 6]['Heart Rate(b/m)'].values

f1, Pxx1 = signal.periodogram(d2_1, fs=1e3)
f2, Pxx2 = signal.periodogram(d2_2, fs=1e3)
f3, Pxx3 = signal.periodogram(d2_3, fs=1e3)

freq1 = f1[np.argmax(Pxx1)]
freq2 = f2[np.argmax(Pxx2)]
freq3 = f3[np.argmax(Pxx3)]

avg_freq = np.mean([freq1, freq2, freq3])

bpm_est = avg_freq * 60
