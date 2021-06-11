# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 00:47:21 2021

@author: 18315
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy.io import loadmat
import ast
import scipy.fft as fft
from construct_dataset import bp_to_category, DataConstructor
from scipy.ndimage.filters import uniform_filter1d
from scipy.optimize import least_squares


#%% Helpers


class DataPipeliner(DataConstructor):
    
    def __init__(self, time_cutoff=10.0, filter_order=2, fs=10, filter_lb=0.8,
                 filter_ub=3.5, scale_a=40, scale_b=170):
        """
        
        Parameters
        ----------
        time_cutoff : TYPE, optional
            Time Cutoff for input signal. The default is 10.0.
            First 10 seconds is "calibration", so we only need everything after
        filter_order : TYPE, optional
            Order of butterworth filter. The default is 2.
        fs : TYPE, optional
            Assumed sampling frequency. The default is 10hz
        filter_lb : TYPE, optional
            Frequency lowerbound for butterworth filter. The default is 0.5hz
        filter_ub : TYPE, optional
            Frequency upperbound for butterworth filter The default is 3.5hz
        scale_a : TYPE, optional
            Rescale the force data lower bound, for the oscillometric method.
            The default is 30.
        scale_b : TYPE, optional
            Rescale the force upper bound, for the oscillometric method.
            The default is 170.

        Returns
        -------
        None.

        """
        self.time_cutoff = time_cutoff
        self.filter_order = filter_order
        self.fs = fs
        self.filter_lb = filter_lb
        self.filter_ub = filter_ub
        self.scale_a = scale_a
        self.scale_b = scale_b
    
    def get_freqs(self, x):
        f = np.abs(np.fft.fft(x))**2
        f = f[len(f)//2:]
        return f
    
    def get_max_freq(self, x):
        return np.max(self.get_freqs(x))
    
    def autoregression_coef(self, x, lag):
        predictors = x[:-lag]
        target = x[lag:]
        model = stats.linregress(predictors, target)
        return model.slope
    
    def feature_extractor(self, ts, num_windows=10):
        features = []
        ts = (ts - ts.mean()) / ts.std()
        # Extract window features
        window_size = int(len(ts)/num_windows)
        windows = []
        for i in range(num_windows):
            start = i * window_size
            end = (i+1) * window_size
            window = ts[start:end]
            windows.append(window)
            
        window_stats = [[np.mean(x), np.std(x), np.min(x), np.max(x), x[-1] - x[0],
                         stats.skew(x), stats.kurtosis(x), self.get_max_freq(x),
                         self.autoregression_coef(x, 1), self.autoregression_coef(x, 2),
                         self.autoregression_coef(x, 3), self.autoregression_coef(x, 4)]
                        for x in windows]
        
        features = np.array(window_stats).reshape(-1)
        return features
    
    def poly_smoother(self, x, y, order=3):
        """

        Parameters
        ----------
        x : float, 1d array-like.
        y : float, 1d array-like
        order : int, optional
            Order of the polynomial smoother. The default is 3.

        Returns
        -------
        A smooth version of y, using polynomial with given order.

        """
        z = np.polyfit(x, y, order)
        p = np.poly1d(z)
        return p(x)
    
    # TODO: document this
    # Helper to process signal inputs
    def process_signal_input(self, time, force, ppg, scale=True):
  
        
        mask = time > self.time_cutoff
        time_ = time[mask]
        ap = force[mask]
        x = ppg[mask]
        x = (x - x.mean()) / x.std()
        if scale:
            a = self.scale_a
            b = self.scale_b
            ap = (b - a) * (ap - ap.min()) / (ap.max() - ap.min()) + a
        
        # Smooth the applied force with polynomial order 3
        smoothed_ap = self.poly_smoother(time_, ap)
        
        # 2nd Order Butterworth filter
        fn = self.fs / 2
        cutoffs = [self.filter_lb/fn, self.filter_ub/fn]
        sos = signal.butter(self.filter_order, cutoffs, btype='band',
                            output='sos')
        filtered_x = signal.sosfilt(sos, x)
        
        # Remove first few points: gets rid of noisy initial state
        filtered_x = filtered_x[10:]
        ap = ap[10:]
        time_ = time_[10:]
        
        assert(len(filtered_x) == len(ap))
        assert(len(ap) == len(time_))
        
        return time_, ap, filtered_x
    
    
    def get_processed_test_data(self, directory):
        # Gets all the data and runs it through the processing pipeline
        data_dict = dict()
        test_data = self.load_test_data(directory)
        #print(test_data)
        for i in range(len(test_data)):
            data_dict[i] = dict()
            
            time_i = test_data.time[i]#.values
            ap_i = test_data.ap[i]#.values
            ppg_i = test_data.ppg[i]#.value
            bp_i = test_data.bp[i]#.values
            cat_i = test_data.bp_cat[i]#.values
            time_, ap_, ppg_ = self.process_signal_input(time_i, ap_i, ppg_i)
            
            data_dict[i]['time'] = time_
            data_dict[i]['ap'] = ap_
            data_dict[i]['ppg'] = ppg_
            data_dict[i]['bp'] = bp_i
            data_dict[i]['bp_cat'] = cat_i
            
        return data_dict
        
      
  
class ResultsCalculator:
    def __init__(self):
        self.prev_systolic_bias = 3.3
        self.prev_systolic_std_dev = 8.8
        self.prev_diastolic_bias = -5.6
        self.prev_diastolic_std_dev = 7.7
        self.prev_error_range = [40, 50]
        self.prev_n = 32
        self.S0 = np.array([[10, 0], [0, 10]])
        self.lam0 = 0.1
        self.df0 = 6
        
        self.prev_mean = np.array([[3.3], [-5.6]])
        self.prev_cov = np.array([[8.8**2, 0], [0, 7.7**2]])
        self.prev_mu = (32/32.1) * self.prev_mean
        self.prev_Sn = self.S0 + self.prev_cov + ((0.1 * 32)/(32.1))*self.prev_mean.dot(self.prev_mean.T)
        self.prev_lam_n = 32.1
        self.prev_df_n = 32 + self.df0
                
    def posterior_update(self, x, n):
        emp_mean = np.mean(x, axis=0).reshape(-1, 1)
        lam_n = self.lam0 + n
        df_n = self.df0 + n

        temp = x.T - emp_mean
        Sn = self.S0 + temp.dot(temp.T) + ((self.lam0 * n) / (lam_n)) * emp_mean.dot(emp_mean.T)
        #print(Sn)
        mu_n = (n / (lam_n)) * emp_mean
        return mu_n, Sn, lam_n, df_n
        
    def posterior_sample(self, num_sample, mu_n, Sn, lam_n, df_n):
        sample_covs = stats.invwishart(df_n, Sn).rvs(size=num_sample)
        sample_means = np.zeros((num_sample, 2))
        for i in range(num_sample):
            c = sample_covs[i]/lam_n
            sample_means[i, :] = stats.multivariate_normal(mu_n.reshape(-1), c).rvs()
            
        return sample_means, sample_covs
        
    def do_AB_test(self, n, x1, x2=None, num_sample=1000):
        # Does an AB test to compare to old results
        if x2 is None:
            mu_n, S_n, lam_n, df_n = self.posterior_update(x1, n)
            means_A, covs_A = self.posterior_sample(num_sample, self.prev_mu,
                                                    self.prev_Sn,
                                                    self.prev_lam_n, 
                                                    self.prev_df_n)
            means_B, covs_B = self.posterior_sample(num_sample, mu_n, S_n, lam_n, df_n)
            
        else:
            mu_n1, S_n1, lam_n1, df_n1 = self.posterior_update(x1, n)
            mu_n2, S_n2, lam_n2, df_n2 = self.posterior_update(x2, n)
            means_A, covs_A = self.posterior_sample(num_sample, mu_n1, S_n1, lam_n1, df_n1)
            means_B, covs_B = self.posterior_sample(num_sample, mu_n2, S_n2, lam_n2, df_n2)
        
        
        # generate the statistics we want
        prob_sys_improve_bias = np.mean(means_A[:,0] > means_B[:,0])
        prob_dia_improve_bias = np.mean(means_A[:,1] > means_B[:,1])
        
        prob_sys_improve_std = np.mean(covs_A[:,0,0] > covs_B[:,0,0])
        prob_dia_improve_std = np.mean(covs_A[:,1,1] > covs_B[:,1,1])
        
        return prob_sys_improve_bias, prob_dia_improve_bias, prob_sys_improve_std, prob_dia_improve_std
        
    def print_compare_results(self):
        mu_sys = self.curr_systolic_bias
        mu_dia = self.curr_diastolic_bias
        sigma_sys = self.curr_systolic_err
        sigma_dia = self.curr_diastolic_err
        
        print("Avg. Systolic Bias: {}".format(mu_sys))
        print("Avg. Diastolic Bias: {}".format(mu_dia))
        print("Systolic Std.Dev: {}".format(sigma_sys))
        print("Diastolic Std.Dev: {}".format(sigma_sys))
        
    def calculate_bias_error(self, y_true, y_pred):
        biases = np.mean(y_true - y_pred, axis=0)
        standard_devs = np.std(y_true - y_pred, axis=0)
        self.curr_N = len(y_pred)
        self.curr_systolic_bias = biases[0]
        self.curr_diastolic_bias = biases[1]
        self.curr_systolic_err = standard_devs[0]
        self.curr_diastolic_err = standard_devs[1]
        
        
def run_baseline(directory):
    
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
    
    def fit_oscillogram(time, x, y):
        
        # For the discrete oscillogram
        peaks, _ = signal.find_peaks(y)
        x = moving_avg(x[peaks], n=10)
        y = moving_avg(y[peaks], n=10)
        
        
    
        x0 = np.ones(5)
        #x0[0] = b
        x0[1] = x0[1] / 2
        x0[2] = np.mean(x)
        x0[3] = np.std(x[:int(len(x) / 2)])
        x0[4] = np.std(x[int(len(x)/2):])
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 100, 100])
        #yhat = model(results, x)
        results = least_squares(residuals, x0,
                                bounds=bounds,
                                args=(x, y))
        yhat = model(results.x, x)
        results = results.x
        a1 = results[0]
        a2 = results[1]
        b1 = results[2]
        b2 = results[3]
        #b3 = results[4]
        #print(results)
    
        
    
        #predict BP
        dia_bp = 0.65 * b1 - 1.54 * (a2/a1) * b2 + 26.2
        mean_bp = 0.68 * b1 - 1.53 * (a2/a1) * b2 + 38.8
        sys_bp = 2.5 * mean_bp - 1.5 * dia_bp
        
        plt.title("Oscillogram: Est. BP: {}/{} mmHg".format(int(sys_bp), int(dia_bp)))
        plt.xlabel("Applied Force")
        plt.ylabel("Blood Volume Oscillation Amplitude")
        plt.scatter(x, y, label="Discrete Oscillogram", alpha=0.70)
        plt.plot(x, yhat, color="darkorange", label="Continuous Estimate")
        plt.legend()
        plt.show()
        
        return sys_bp, dia_bp
        
    ## BASELINE EXPERIMENT ##
    pipeliner = DataPipeliner()
    processed_data = pipeliner.get_processed_test_data(directory)
    y_true = []
    y_pred = []
    y_cat_true = []
    y_cat_pred = []
    for i in range(len(processed_data)):
        time = processed_data[i]['time']
        ppg = processed_data[i]['ppg']
        ap = processed_data[i]['ap']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat']
        sys_bp, dia_bp = fit_oscillogram(time, ap, ppg)
        
        if sys_bp > 180 or dia_bp > 180 or sys_bp < 50 or dia_bp < 20:
            continue
        
        bp_pred = np.array([sys_bp, dia_bp])
        bp_cat_pred = bp_to_category([sys_bp, dia_bp])
        y_true.append(bp)
        y_pred.append(bp_pred)
        y_cat_true.append(bp_cat)
        y_cat_pred.append(bp_cat_pred)
        
    # Obtain Results
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_cat_true = np.array(y_cat_true)
    y_cat_pred = np.array(y_cat_pred)        
    
    calculator = ResultsCalculator()
    biases = y_true - y_pred
    #print(biases.mean(axis=0))
    mu_n, S_n, lam_n, df_n = calculator.posterior_update(biases, len(biases))
    mean_samples, cov_samples = calculator.posterior_sample(1000, mu_n, S_n, lam_n, df_n)
    #results = calculator.do_AB_test(len(biases), biases)


    return mean_samples, cov_samples, y_cat_true, y_cat_pred

        
    
    
    


#%%
# RUN EXPERIMENTS
directory = r"C:\Users\18315\Documents\Projects\SmartBP\data"


def scores(fp, fn, tp, tn):
    f = tp / (tp + (0.5 * (fp + fn)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return f, precision, recall


def compute_prob_accuracy(trues, preds):
    a = 1
    b = 1
    num_correct = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    n = 0
    for i in range(len(trues)):
        y_true = trues[i]
        y_pred = preds[i]
        
        num_correct += y_true == y_pred
        pos_true = y_true > 1
        pos_pred = y_pred > 1
        fp += y_pred and not pos_true
        tp += y_pred and pos_true
        fn += not y_pred and pos_true
        tn += not y_pred and not pos_true
        n += 1
    f, p, r = scores(fp, fn, tp, tn)
    return num_correct/n, f, p, r

#%% Baseline

np.random.seed(42)

results = {}
results['Method'] = ['Baseline', 'RF', 'KNN', 'RankBP', 'RankBP2']
results['Systolic Bias'] = []
results['Diastolic Bias'] = []
results['Systolic Std.Dev'] = []
results['Diastolic Std.Dev'] = []
results['Precision'] = []
results['Recall'] = []
results['F1'] = []

means1, covs1, cats1, cat_preds1 = run_baseline(directory)
acc, f1_score, precision, recall = compute_prob_accuracy(cats1, cat_preds1)

bias = means1.mean(axis=0)
sigma1 = np.sqrt(covs1[:,0,0].mean())
sigma2 = np.sqrt(covs1[:,1,1].mean())        
results['Systolic Bias'].append(bias[0])
results['Diastolic Bias'].append(bias[1])
results['Systolic Std.Dev'].append(sigma1)
results['Diastolic Std.Dev'].append(sigma2)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1'].append(f1_score)

        
        
#%%%# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

def random_forest_experiments(directory):
    np.random.seed(42)
    ## Random Forest EXPERIMENT ##
    pipeliner = DataPipeliner()
    X_train = []
    y_sys_train = []
    y_dia_train = []
    y_label_train = []
    
    num_resample = 201
    processed_data = pipeliner.load_pretrain_data()
    for i in range(len(processed_data)):
        #time = processed_data[i]['time']
        ppg = processed_data[i]['ppg']
        #ap = processed_data[i]['ap']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat'] 
        ppg = signal.resample(ppg, num_resample)[1:]
        
        feature_vec = pipeliner.feature_extractor(ppg)
        X_train.append(feature_vec)
        y_sys_train.append(bp[0])
        y_dia_train.append(bp[1])
        y_label_train.append(bp_cat)
        
    X_test = []
    y_sys_test = []
    y_dia_test = []
    y_label_test = []
        
    processed_data = pipeliner.get_processed_test_data(directory)
    for i in range(len(processed_data)):
        ppg = processed_data[i]['ppg']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat']
        
        feature_vec = pipeliner.feature_extractor(ppg)
        #print(feature_vec.shape)
        #feature_vec1 = np.concatenate([feature_vec, np.array([1])])
        #feature_vec2 = np.concatenate([feature_vec, np.array([0])])
        X_test.append(feature_vec)
        y_sys_test.append(bp[0])
        y_dia_test.append(bp[1])
        y_label_test.append(bp_cat)
        
    #make everything an array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_sys_train = np.array(y_sys_train)
    y_sys_test = np.array(y_sys_test)
    y_dia_train = np.array(y_dia_train)
    y_dia_test = np.array(y_dia_test)
    y_label_test = np.array(y_label_test)
    
    #LOOCV
    y1 = []
    y2 = []
    for i in range(len(processed_data)):
        x_test = X_test[i].reshape(1, -1)
        y_sys = y_sys_test[i]
        y_dia= y_dia_test[i]
        label = y_label_test[i]
        y_test = np.array([y_sys, y_dia])
        train_idx = np.array([j for j in range(len(X_test))
                              if j != i])
        
        X__ = np.concatenate([X_train, X_test[train_idx]], axis=0)
        y__1 = np.concatenate([y_sys_train, y_sys_test[train_idx]], axis=0)
        y__2 = np.concatenate([y_dia_train, y_dia_test[train_idx]], axis=0)
        
        
        rf1 = RandomForestRegressor()
        rf2 = RandomForestRegressor()
        rf1.fit(X__, y__1)
        rf2.fit(X__, y__2)
        
        y1_ = rf1.predict(x_test)
        y2_ = rf2.predict(x_test)
        y1.append(y1_)
        y2.append(y2_)
        
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    # Obtain Results
    y_true = np.concatenate([y_sys_test.reshape(-1, 1), y_dia_test.reshape(-1, 1)], axis=1)
    y_pred = np.concatenate([y1.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)
    y_cat_true = np.array(y_label_test)
    y_cat_pred = np.array([bp_to_category(y) for y in y_pred])        
    
    calculator = ResultsCalculator()
    biases = y_true - y_pred
    #print(biases.mean(axis=0))
    mu_n, S_n, lam_n, df_n = calculator.posterior_update(biases, len(biases))
    mean_samples, cov_samples = calculator.posterior_sample(1000, mu_n, S_n, lam_n, df_n)
    #results = calculator.do_AB_test(len(biases), biases)


    return mean_samples, cov_samples, y_cat_true, y_cat_pred

means2, covs2, cats2, cat_preds2 = random_forest_experiments(directory)
acc, f1_score, precision, recall = compute_prob_accuracy(cats2, cat_preds2)

bias = means2.mean(axis=0)
sigma1 = np.sqrt(covs2[:,0,0].mean())
sigma2 = np.sqrt(covs2[:,1,1].mean())        
results['Systolic Bias'].append(bias[0])
results['Diastolic Bias'].append(bias[1])
results['Systolic Std.Dev'].append(sigma1)
results['Diastolic Std.Dev'].append(sigma2)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1'].append(f1_score)


#%%



#%% KNN

from sklearn.neighbors import KNeighborsRegressor

def KNN_experiments(directory):
    np.random.seed(42)
    pipeliner = DataPipeliner()
    X_train = []
    y_sys_train = []
    y_dia_train = []
    y_label_train = []
    
    num_resample = 201
    processed_data = pipeliner.load_pretrain_data()
    for i in range(len(processed_data)):
        ppg = processed_data[i]['ppg']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat'] 
        ppg = signal.resample(ppg, num_resample)[1:]
        
        feature_vec = pipeliner.feature_extractor(ppg)
        X_train.append(feature_vec)
        y_sys_train.append(bp[0])
        y_dia_train.append(bp[1])
        y_label_train.append(bp_cat)
        
    X_test = []
    y_sys_test = []
    y_dia_test = []
    y_label_test = []
        
    processed_data = pipeliner.get_processed_test_data(directory)
    for i in range(len(processed_data)):
        ppg = processed_data[i]['ppg']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat']
        
        feature_vec = pipeliner.feature_extractor(ppg)

        X_test.append(feature_vec)
        y_sys_test.append(bp[0])
        y_dia_test.append(bp[1])
        y_label_test.append(bp_cat)
        
    #make everything an array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_sys_train = np.array(y_sys_train)
    y_sys_test = np.array(y_sys_test)
    y_dia_train = np.array(y_dia_train)
    y_dia_test = np.array(y_dia_test)
    y_label_test = np.array(y_label_test)
    
    #LOOCV
    y1 = []
    y2 = []
    for i in range(len(processed_data)):
        x_test = X_test[i].reshape(1, -1)
        y_sys = y_sys_test[i]
        y_dia= y_dia_test[i]
        label = y_label_test[i]
        y_test = np.array([y_sys, y_dia])
        train_idx = np.array([j for j in range(len(X_test))
                              if j != i])
        
        X__ = np.concatenate([X_train, X_test[train_idx]], axis=0)
        X_min = X__.min(axis=0)
        X_max = X__.max(axis=0)
        X__ = (X__ - X_min) / (X_max - X_min)
        x_test = (x_test - X_min) / (X_max - X_min)
        y__1 = np.concatenate([y_sys_train, y_sys_test[train_idx]], axis=0)
        y__2 = np.concatenate([y_dia_train, y_dia_test[train_idx]], axis=0)
        
        
        knn1 = KNeighborsRegressor(10)
        knn2 = KNeighborsRegressor(10)
        knn1.fit(X__, y__1)
        knn2.fit(X__, y__2)
        
        y1_ = knn1.predict(x_test)
        y2_ = knn2.predict(x_test)
        y1.append(y1_)
        y2.append(y2_)
        
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    # Obtain Results
    y_true = np.concatenate([y_sys_test.reshape(-1, 1), y_dia_test.reshape(-1, 1)], axis=1)
    y_pred = np.concatenate([y1.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)
    y_cat_true = np.array(y_label_test)
    y_cat_pred = np.array([bp_to_category(y) for y in y_pred])        
    
    calculator = ResultsCalculator()
    biases = y_true - y_pred
    #print(biases.mean(axis=0))
    mu_n, S_n, lam_n, df_n = calculator.posterior_update(biases, len(biases))
    mean_samples, cov_samples = calculator.posterior_sample(1000, mu_n, S_n, lam_n, df_n)
    #results = calculator.do_AB_test(len(biases), biases)


    return mean_samples, cov_samples, y_cat_true, y_cat_pred

means3, covs3, cats3, cat_preds3 = KNN_experiments(directory)
acc, f1_score, precision, recall = compute_prob_accuracy(cats3, cat_preds3)

bias = means3.mean(axis=0)
sigma1 = np.sqrt(covs3[:,0,0].mean())
sigma2 = np.sqrt(covs3[:,1,1].mean())        
results['Systolic Bias'].append(bias[0])
results['Diastolic Bias'].append(bias[1])
results['Systolic Std.Dev'].append(sigma1)
results['Diastolic Std.Dev'].append(sigma2)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1'].append(f1_score)




#%% 

#RANK BP
def compute_V(q, x_plus, x_neg):
    diff = x_plus - x_neg
    V = np.array([q[i]*diff for i in range(len(q))])
    return V

def compute_tau(q, x_plus, x_neg, W, V, C=0.10):
    l = loss(q, x_plus, x_neg, W)
    tau = np.min([C, l / np.linalg.norm(V)**2])
    return tau

def loss(q, x_plus, x_neg, W):
    distance = 1 - q.dot(W).dot(x_plus) + q.dot(W).dot(x_neg)
    return np.max([0, distance])


def update_step(q, x_plus, x_neg, W, C=0.10):
    V = compute_V(q, x_plus, x_neg)
    tau = compute_tau(q, x_plus, x_neg, W, V, C)
    W_new = W + tau*V
    return W_new

    
def draw_triplet(X_single, labels):
    while True:
        i = np.random.choice(len(X_single))
        j = np.random.choice(len(X_single))
        k = np.random.choice(len(X_single))
        
        q = X_single[i]
        q_label = labels[i]
        
        x1 = X_single[j]
        x1_label = labels[j]
    
        x2 = X_single[k]
        x2_label = labels[k]
        
        dist1 = np.linalg.norm(q_label - x1_label)
        dist2 = np.linalg.norm(q_label - x2_label)
        #dist1 = np.abs(q_label - x1_label)
        #dist2 = np.abs(q_label - x2_label)
        if dist1 < dist2:
            x_plus = x1
            x_neg = x2
            #print("q label: {}, plus label: {}, neg label: {}".format(q_label, x1_label, x2_label))
            return q, x_plus, x_neg
        elif dist2 < dist1:
            x_plus = x2
            x_neg = x1
            #print("q label: {}, plus label: {}, neg label: {}".format(q_label, x2_label, x1_label))
            return q, x_plus, x_neg
        else:
            continue
   
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def rank_experiments(directory):
    np.random.seed(42)
    pipeliner = DataPipeliner()
    X_train = []
    y_sys_train = []
    y_dia_train = []
    y_label_train = []
    
    num_resample = 201
    processed_data = pipeliner.load_pretrain_data()
    for i in range(len(processed_data)):
        #time = processed_data[i]['time']
        ppg = processed_data[i]['ppg']
        #ap = processed_data[i]['ap']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat'] 
        ppg = signal.resample(ppg, num_resample)[1:]
        
        feature_vec = pipeliner.feature_extractor(ppg)
        X_train.append(feature_vec)
        y_sys_train.append(bp[0])
        y_dia_train.append(bp[1])
        y_label_train.append(bp_cat)
        
    X_test = []
    y_sys_test = []
    y_dia_test = []
    y_label_test = []
        
    processed_data = pipeliner.get_processed_test_data(directory)
    for i in range(len(processed_data)):
        ppg = processed_data[i]['ppg']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat']
        
        feature_vec = pipeliner.feature_extractor(ppg)
        #print(feature_vec.shape)
        #feature_vec1 = np.concatenate([feature_vec, np.array([1])])
        #feature_vec2 = np.concatenate([feature_vec, np.array([0])])
        X_test.append(feature_vec)
        y_sys_test.append(bp[0])
        y_dia_test.append(bp[1])
        y_label_test.append(bp_cat)
        
    #make everything an array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_sys_train = np.array(y_sys_train)
    y_sys_test = np.array(y_sys_test)
    y_dia_train = np.array(y_dia_train)
    y_dia_test = np.array(y_dia_test)
    y_label_test = np.array(y_label_test)
    y_label_train = np.array(y_label_train)
    
    #LOOCV
    y1 = []
    y2 = []
    for i in range(len(processed_data)):
        x_test = X_test[i].reshape(1, -1)
        y_sys = y_sys_test[i]
        y_dia= y_dia_test[i]
        label = y_label_test[i]
        y_test = np.array([y_sys, y_dia])
        train_idx = np.array([j for j in range(len(X_test))
                              if j != i])
        
        X__ = np.concatenate([X_train, X_test[train_idx]], axis=0)
        y__1 = np.concatenate([y_sys_train, y_sys_test[train_idx]], axis=0)
        y__2 = np.concatenate([y_dia_train, y_dia_test[train_idx]], axis=0)
        
        X_min = X__.min(axis=0)
        X_max = X__.max(axis=0)
        X__ = (X__ - X_min) / (X_max - X_min)
        x_test = (x_test - X_min) / (X_max - X_min)
        labels__ = np.concatenate([y_label_train, y_label_test[train_idx]])
        labels__1 = np.concatenate([y__1.reshape(-1, 1), y__2.reshape(-1, 1)], axis=1)
        
        W = np.eye(X__.shape[1])
        C = 0.10
        # Train Rank BP
        for t in range(5000):
            q, x_plus, x_neg = draw_triplet(X__, labels__1)
            W = update_step(q, x_plus, x_neg, W, C)
            #W = 0.5 * (W.T + W)
            
        #project matrix
        W = nearestPD(W)
        x_test = x_test.reshape(-1)
        neighbors = [(x_test.dot(W).dot(X__[i]), i) for i in range(len(X__))]
        neighbors = sorted(neighbors, key=lambda t : t[0], reverse=True)
        top_ids = np.array([t[1] for t in neighbors[:10]])
        y__1 = np.mean(y__1[top_ids])
        y__2 = np.mean(y__2[top_ids])
        y1.append(y__1)
        y2.append(y__2)
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    # Obtain Results
    #print(y1)
    y_true = np.concatenate([y_sys_test.reshape(-1, 1), y_dia_test.reshape(-1, 1)], axis=1)
    y_pred = np.concatenate([y1.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)
    y_cat_true = np.array(y_label_test)
    y_cat_pred = np.array([bp_to_category(y) for y in y_pred])        
    
    calculator = ResultsCalculator()
    biases = y_true - y_pred
    #print(biases.mean(axis=0))
    mu_n, S_n, lam_n, df_n = calculator.posterior_update(biases, len(biases))
    mean_samples, cov_samples = calculator.posterior_sample(1000, mu_n, S_n, lam_n, df_n)
    #results = calculator.do_AB_test(len(biases), biases)


    return mean_samples, cov_samples, y_cat_true, y_cat_pred

means4, covs4, cats4, cat_preds4 = rank_experiments(directory)
acc, f1_score, precision, recall = compute_prob_accuracy(cats4, cat_preds4)

bias = means4.mean(axis=0)
sigma1 = np.sqrt(covs4[:,0,0].mean())
sigma2 = np.sqrt(covs4[:,1,1].mean())        
results['Systolic Bias'].append(bias[0])
results['Diastolic Bias'].append(bias[1])
results['Systolic Std.Dev'].append(sigma1)
results['Diastolic Std.Dev'].append(sigma2)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1'].append(f1_score)

#%%%


def rank_experiments2(directory):
    np.random.seed(42)
    pipeliner = DataPipeliner()
    X_train = []
    y_sys_train = []
    y_dia_train = []
    y_label_train = []
    
    num_resample = 201
    processed_data = pipeliner.load_pretrain_data()
    for i in range(len(processed_data)):
        #time = processed_data[i]['time']
        ppg = processed_data[i]['ppg']
        #ap = processed_data[i]['ap']
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat'] 
        ppg = signal.resample(ppg, num_resample)[1:]
        ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
        X_train.append(ppg)
        y_sys_train.append(bp[0])
        y_dia_train.append(bp[1])
        y_label_train.append(bp_cat)
        
    X_test = []
    y_sys_test = []
    y_dia_test = []
    y_label_test = []
        
    processed_data = pipeliner.get_processed_test_data(directory)
    for i in range(len(processed_data)):
        ppg = processed_data[i]['ppg']
        ppg = signal.resample(ppg, num_resample)[1:]
        ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
        bp = np.array(processed_data[i]['bp'])
        bp_cat = processed_data[i]['bp_cat']

        X_test.append(ppg)
        y_sys_test.append(bp[0])
        y_dia_test.append(bp[1])
        y_label_test.append(bp_cat)
        
    #make everything an array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_sys_train = np.array(y_sys_train)
    y_sys_test = np.array(y_sys_test)
    y_dia_train = np.array(y_dia_train)
    y_dia_test = np.array(y_dia_test)
    y_label_test = np.array(y_label_test)
    y_label_train = np.array(y_label_train)
    
    
    #LOOCV
    y1 = []
    y2 = []
    for i in range(len(processed_data)):
        x_test = X_test[i]
        y_sys = y_sys_test[i]
        y_dia= y_dia_test[i]
        label = y_label_test[i]
        y_test = np.array([y_sys, y_dia])
        train_idx = np.array([j for j in range(len(X_test))
                              if j != i])
        
        X__ = np.concatenate([X_train, X_test[train_idx]], axis=0)
        y__1 = np.concatenate([y_sys_train, y_sys_test[train_idx]], axis=0)
        y__2 = np.concatenate([y_dia_train, y_dia_test[train_idx]], axis=0)

        labels__ = np.concatenate([y_label_train, y_label_test[train_idx]])
        labels__1 = np.concatenate([y__1.reshape(-1, 1), y__2.reshape(-1, 1)], axis=1)
        W = np.eye(X__.shape[1])
        C = 0.10
        # Train Rank BP
        for t in range(2000):
            q, x_plus, x_neg = draw_triplet(X__, labels__1)
            W = update_step(q, x_plus, x_neg, W, C)
            #W = 0.5 * (W.T + W)
            
        #project matrix
        W = nearestPD(W)
        x_test = x_test.reshape(-1)
        neighbors = [(x_test.dot(W).dot(X__[i]), i) for i in range(len(X__))]
        neighbors = sorted(neighbors, key=lambda t : t[0], reverse=True)
        top_ids = np.array([t[1] for t in neighbors[:5]])
        y__1 = np.mean(y__1[top_ids])
        y__2 = np.mean(y__2[top_ids])
        y1.append(y__1)
        y2.append(y__2)
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    # Obtain Results
    y_true = np.concatenate([y_sys_test.reshape(-1, 1), y_dia_test.reshape(-1, 1)], axis=1)
    #print(y_true)
    y_pred = np.concatenate([y1.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)
    #print(y_pred)
    y_cat_true = np.array(y_label_test)
    y_cat_pred = np.array([bp_to_category(y) for y in y_pred])        
    
    calculator = ResultsCalculator()
    biases = y_true - y_pred
    #print(biases.mean(axis=0))
    mu_n, S_n, lam_n, df_n = calculator.posterior_update(biases, len(biases))
    mean_samples, cov_samples = calculator.posterior_sample(1000, mu_n, S_n, lam_n, df_n)
    #results = calculator.do_AB_test(len(biases), biases)


    return mean_samples, cov_samples, y_cat_true, y_cat_pred

means5, covs5, cats5, cat_preds5 = rank_experiments2(directory)
acc, f1_score, precision, recall = compute_prob_accuracy(cats5, cat_preds5)

bias = means5.mean(axis=0)
sigma1 = np.sqrt(covs5[:,0,0].mean())
sigma2 = np.sqrt(covs5[:,1,1].mean())        
results['Systolic Bias'].append(bias[0])
results['Diastolic Bias'].append(bias[1])
results['Systolic Std.Dev'].append(sigma1)
results['Diastolic Std.Dev'].append(sigma2)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1'].append(f1_score)

#%% Generate a latex table

results = pd.DataFrame(results)

print(results.to_latex(index=False))

n = 400
plt.title("Posterior Distribution of Systolic Biases")
plt.hlines(0, xmin=0, xmax=n, colors='black', linestyles='--', label='Zero')
plt.hlines(means1.mean(axis=0)[0], xmin=0, xmax=n, colors='darkblue', linestyles='--')
plt.hlines(means2.mean(axis=0)[0], xmin=0, xmax=n, colors='red', linestyles='--')
plt.hlines(means5.mean(axis=0)[0], xmin=0, xmax=n, colors='green', linestyles='--')
plt.scatter(np.arange(n), means1[:n,0], alpha=0.3, label='Baseline')
plt.scatter(np.arange(n), means2[:n,0], alpha=0.3, label='RandomForest')
plt.scatter(np.arange(n), means5[:n,0], alpha=0.3, label='RankBP2')
plt.legend()
plt.show()

n = 400
plt.title("Posterior Distribution of Diastolic Biases")
plt.hlines(0, xmin=0, xmax=n, colors='black', linestyles='--', label='Zero')
plt.hlines(means1.mean(axis=0)[1], xmin=0, xmax=n, colors='darkblue', linestyles='--')
plt.hlines(means2.mean(axis=0)[1], xmin=0, xmax=n, colors='red', linestyles='--')
plt.hlines(means5.mean(axis=0)[1], xmin=0, xmax=n, colors='green', linestyles='--')
plt.scatter(np.arange(n), means1[:n,1], alpha=0.3, label='Baseline')
plt.scatter(np.arange(n), means2[:n,1], alpha=0.3, label='RandomForest')
plt.scatter(np.arange(n), means5[:n,1], alpha=0.3, label='RankBP2')
plt.legend()
plt.show()


#%% get some extra statistics


lb1 = np.quantile(means1, 0.05, axis=0)
ub1 = np.quantile(means1, 0.95, axis=0)

lb2 = np.quantile(means2, 0.05, axis=0)
ub2 = np.quantile(means2, 0.95, axis=0)

lb3 = np.quantile(means3, 0.05, axis=0)
ub3 = np.quantile(means3, 0.95, axis=0)

lb4 = np.quantile(means4, 0.05, axis=0)
ub4 = np.quantile(means4, 0.95, axis=0)

lb5 = np.quantile(means5, 0.05, axis=0)
ub5 = np.quantile(means5, 0.95, axis=0)


p_imp2 = (np.abs(0 - means2) < np.abs(0 - means1)).mean(axis=0)
p_imp3 = (np.abs(0 - means3) < np.abs(0 - means1)).mean(axis=0)
p_imp4 = (np.abs(0 - means4) < np.abs(0 - means1)).mean(axis=0)
p_imp5 = (np.abs(0 - means5) < np.abs(0 - means1)).mean(axis=0)

p_o2 = (np.linalg.norm(means2,axis=1) < np.linalg.norm(means1, axis=1)).mean()
p_o3 = (np.linalg.norm(means3,axis=1) < np.linalg.norm(means1, axis=1)).mean()
p_o4 = (np.linalg.norm(means4, axis=1) < np.linalg.norm(means1, axis=1)).mean()
p_o5 = (np.linalg.norm(means5, axis=1) < np.linalg.norm(means1, axis=1)).mean()

results2 = {}
results2['model'] = results.Method
results2['Sys. Bias Cred. Interval'] = [[np.round(lb1[0], 3), np.round(ub1[0], 3)],
                                   [np.round(lb2[0], 3), np.round(ub2[0], 3)],
                                   [np.round(lb3[0], 3), np.round(ub3[0], 3)],
                                   [np.round(lb4[0], 3), np.round(ub4[0], 3)],
                                   [np.round(lb5[0], 3), np.round(ub5[0], 3)]]

results2['Dia. Bias Cred. Interval'] = [[np.round(lb1[1], 3), np.round(ub1[1], 3)],
                                   [np.round(lb2[1], 3), np.round(ub2[1], 3)],
                                   [np.round(lb3[1], 3), np.round(ub3[1], 3)],
                                   [np.round(lb4[1], 3), np.round(ub4[1], 3)],
                                   [np.round(lb5[0], 3), np.round(ub5[0], 3)]]

results2['Sys. Prob. Improvement'] = [0,
                                      np.round(p_imp2[0], 3),
                                      np.round(p_imp3[0], 3),
                                     np.round(p_imp4[0], 3),
                                     np.round(p_imp5[0], 3)]

results2['Dia. Prob. Improvement'] = [0,
                                      np.round(p_imp2[1], 3),
                                      np.round(p_imp3[1], 3),
                                     np.round(p_imp4[1], 3),
                                     np.round(p_imp5[1], 3)]

results2['Prob. Overall Improve'] = [0, p_o2, p_o3, p_o4, p_o5]



results2 = pd.DataFrame(results2)
print(results2.to_latex(index=False))




















