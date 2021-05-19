# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:41:19 2021

@author: 18315
"""

#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares, leastsq
from scipy.ndimage.filters import uniform_filter1d
import heartpy as hp
from scipy.fft import fft, ifft
from statsmodels.tsa.seasonal import seasonal_decompose




#%% Helper functions

# Wrapper around sklearn's linear regression and polyfeatures to work
# as a polynomial smoother. May not be the fastest solution but simple
def poly_smoother(x, y, degree=3):
    ols_smoother = LinearRegression()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    if len(x.shape) == 1:
        x_d = poly.fit_transform(x.reshape(-1, 1))
    else:
        x_d = poly.fit_transform(x)
        
    ols_smoother.fit(x_d, y)
    smoothed_y = ols_smoother.predict(x_d)
    return smoothed_y
    

# Function to compute the average of the reciprocral of the peak-to-peak
# time interval (in seconds)
def avg_beats_per_sec(time, ppg, fs):
    
    peak_idxs, _ = signal.find_peaks(ppg)
    time_peaks = time[peak_idxs]
    
    # computes the peak to peak intervals
    average = np.mean( 1 / (time_peaks[1:] - time_peaks[:-1]))
    #avg_beats_per_sec = 1 / np.median(time_peaks[1:] - time_peaks[:-1])
    return average
        

def spectral_bpm(time, ap, ppg, tp=150):
    #get the time measurements from 40hgmm to termination pressure
    mask = (ap >= 40)
    time0 = time[mask]
    ppg0 = ppg[mask]
    
    # estimate the average interval between samples
    #delta = np.mean(time0[1:] - time0[:-1])
    #fs = 1 / delta  # estimate of the frequency
    fs = 5.0
    f, Pxx = signal.welch(ppg0, fs)
    

    
    # we only care about frequencies in this range
    mask = (f >= 0.5) & (f <= 3)
    
    peak_freq = f[mask][np.argmax(Pxx[mask])]  # largest frequency in hertz
    
    peaks, _ = signal.find_peaks(Pxx[mask])
    print(Pxx[peaks])
    
    plt.title("Spectrum of Blood Volume")
    plt.ylabel("Power Density [V**2/Hz]")
    plt.xlabel("Freq [Hz]")
    plt.plot(f[mask], Pxx[mask])
    plt.scatter(f[mask][peaks], Pxx[mask][peaks], c="darkorange")
    plt.show()
    
    
    print(peak_freq)
    return peak_freq * 60

# Helper function to compute a moving average using n data points
# This should be the fastest and most stable. Wrapper for readability
def moving_avg(x, n=3):
    smoothed_x = uniform_filter1d(x, n, mode='reflect')
    return smoothed_x

def model(variables, x):
    a1 = variables[0]
    a2 = variables[1]
    b1 = variables[2]
    b2 = variables[3]
    b3 = variables[4]
    
    yhat = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < b1:
            yhat[i] = a2 + (a1 - a2) * np.exp(-0.50 * ((x[i] - b1)/b2)**2)
        else:
            yhat[i] = a1 * np.exp(-0.50 * ((x[i] - b1)/b3)**2)
    
    return yhat
    
# The objective function: computes the residuals of the oscillogram
# using the parametric model described in the paper
def residuals(variables, x, y):
    return y - model(variables, x)

def fit_oscillogram(time, ap, ppg):
    mask = (ap > 40)
    time0 = time[mask]
    ap0 = ap[mask]
    ppg0 = ppg[mask]
    demeaned_ppg = ppg0 - np.mean(ppg0)
    #demeaned_ppg = demeaned_ppg[1:] - demeaned_ppg[:-1]
    results = seasonal_decompose(ppg0, model='additive',freq=5)
    #print(results.seasonal.shape)
    #print(ap0.shape)
    smooth_ap = poly_smoother(time0, ap0)
    demeaned_ppg = results.seasonal + results.resid
    
    # For the discrete oscillogram
    peaks, _ = signal.find_peaks(demeaned_ppg)
    ap_peaks = moving_avg(smoothed_ap[peaks])
    ppg_peaks = moving_avg(demeaned_ppg[peaks])
    
    
    # initial estimates
    y_tp = ap_peaks.max()
    x0 = np.ones(5)
    x0[1] = x0[1] / 2
    x0[2] = np.mean(x)
    x0[3] = np.std(x[:int(len(x) / 2)])
    x0[4] = np.std(x[int(len(x)/2):])
    bounds = ([y_tp / 0.50, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 100, 100])
    #results, flag = leastsq(residuals, x0, args=(x, smoothed_ppg_peaks))
    #yhat = model(results, x)
    results = least_squares(residuals, x0, args=(ap_peaks, ppg_peaks))
    yhat = model(results.x, ap_peaks)
    results = results.x
    a1 = results[0]
    a2 = results[1]
    b1 = results[2]
    b2 = results[3]
    b3 = results[4]
    print(results)

    #predict BP
    dia_bp = 0.65 * b1 - 1.54 * (a2/a1) * b2 + 26.2
    mean_bp = 0.68 * b1 - 1.53 * (a2/a1) * b2 + 38.8
    sys_bp = 2.5 * mean_bp - 1.5 * dia_bp
    
    print("dia_bp: {}".format(dia_bp))
    print("sys_bp: {}".format(sys_bp))
        
    
    plt.title("Oscillogram")
    plt.xlabel("Applied Force")
    plt.ylabel("Blood Volume Oscillation Amplitude")
    plt.scatter(ap_peaks, ppg_peaks, label="Discrete Oscillogram")
    plt.plot(ap_peaks, yhat, color="red", label="Continuous Estimate")
    plt.legend()
    plt.show()
    
    
    
    





#%% READ IN DATA

# Hard coded for now. Figure out the IO stuff later.
path = r"C:\Users\18315\Documents\Projects\SmartBP\forcedata6-84bpm-bp134by87mmhg.txt"
#path = r"C:\Users\18315\Documents\Projects\SmartBP\forcedata7-78bpm-bp125by85mmhg.txt"
#path = r"C:\Users\18315\Documents\Projects\SmartBP\forcedata8-91bpm-bp127by88mmhg.txt"
data = pd.read_csv(path)

# Note: 200 datapts/40 sec = 5 datapt/1 sec
# Remove first 10 seconds : Remove first 50 datapts

# 30sec of time
time = data['time'][-150:].values
time = time - time[0]



# 30sec applied pressure
ap = data['force'][-150:].values

# 30sec of ppg sensor
ppg = data['ppg0'][-150:].values
mean_ppg = np.mean(ppg)
demeaned_ppg = ppg - mean_ppg

#trend = LinearRegression()
#trend.fit(time.reshape(-1,1), demeaned_ppg)
#ppg_hat = trend.predict(time.reshape(-1, 1))
#detrended_ppg = demeaned_ppg - ppg_hat
detrended_ppg = demeaned_ppg[1:] - demeaned_ppg[:-1]



#%% DATA PROCESSING

# smoothing the applied pressure with a 3rd order polynomial fit

# Note: This is just linear regression with polynomial features,
# and scikit learn is probably more numerically stable, so I'm using this
# instead of numpy. The numpy and sklearn residuals are nearly identitical



# Plots the AP vs time and its smoothed version, for a sanity check
#plt.xlabel("Time")
#plt.ylabel("Pressure")
#plt.plot(time, ap, '-', time, smoothed_ap)

avg_ap = moving_avg(ap, n=5)

# TODO: get blood volume data and convert it to a heart rate measurement
avg_beats = avg_beats_per_sec(time[1:], demeaned_ppg, 5)
avg_hr = 60 * avg_beats



peak_idxs, _ = signal.find_peaks(demeaned_ppg)
ppg_peaks = demeaned_ppg[peak_idxs]
#ap_peaks = ap[1:][peak_idxs]
smoothed_ap = poly_smoother(time, ap)
ap_peaks = smoothed_ap[1:][peak_idxs]

# smooth with a 3 point moving average
smoothed_ppg_peaks = moving_avg(ppg_peaks, 3)
smoothed_ap_peaks = moving_avg(ap_peaks, 3)
mean_ap = np.mean(smoothed_ap_peaks)
sd_ap = np.std(smoothed_ap_peaks)
#x = (smoothed_ap_peaks - mean_ap) / sd_ap
x = smoothed_ap_peaks
y = smoothed_ppg_peaks
plt.scatter(x, smoothed_ppg_peaks)
plt.plot(x, smoothed_ppg_peaks)
plt.show()



results = seasonal_decompose(ppg, model="additive", freq=5)


peaks, _ = signal.find_peaks(ppg)

plt.title("Blood Volume Time Series")
plt.ylabel("Blood Volume")
plt.xlabel("Time (sec)")
plt.plot(time, ppg)
plt.scatter(time[peaks], ppg[peaks], color="darkorange")
plt.legend(labels=["HR Truth = 78"])
plt.show()


fit_oscillogram(time, ap, ppg)


