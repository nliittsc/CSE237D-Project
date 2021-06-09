import matplotlib.pyplot as plt
import matplotlib.animation as animation
from arduinoIO import setup, close, receive_message
from time import time
import numpy as np
import serial
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares



#%%


# Helper function to compute a moving average using n data points
# This should be the fastest and most stable. Wrapper for readability
def moving_avg(x, n=3):
    smoothed_x = uniform_filter1d(x, n, mode='reflect')
    return smoothed_x






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
    mask = time > 10
    mask = mask * (ap > 40)
    time0 = time[mask]
    ap0 = ap[mask]
    ppg0 = ppg[mask]
    ppg0 = (ppg0 - ppg0.mean()) / ppg0.std()
    #demeaned_ppg = ppg0 - np.mean(ppg0)
    a = 40
    b = 170
    ap0 = (b - a) * ((ap0 - ap0.min()) / (ap0.max() - ap0.min())) + a
    smooth_ap = poly_smoother(time0, ap0)

    
    fs = 10
    fn = fs / 2
    order = 2
    lb = 0.5
    ub = 3.5
    cutoffs = [lb / fn, ub/ fn]
    sos = signal.butter(order, cutoffs, btype='band', output='sos')
    #sos = signal.butter(order, lb/fn, btype='high', output='sos')
    x = smooth_ap[10:]
    y = signal.sosfilt(sos, ppg0)[10:]
    
      
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
    print(results)

    #predict BP
    dia_bp = 0.65 * b1 - 1.54 * (a2/a1) * b2 + 26.2
    mean_bp = 0.68 * b1 - 1.53 * (a2/a1) * b2 + 38.8
    sys_bp = 2.5 * mean_bp - 1.5 * dia_bp
    
    print("dia_bp: {}".format(dia_bp))
    print("sys_bp: {}".format(sys_bp))
        
    
    plt.title("Oscillogram: Est. BP: {}/{} mmHg".format(int(sys_bp), int(dia_bp)))
    plt.xlabel("Applied Force")
    plt.ylabel("Blood Volume Oscillation Amplitude")
    plt.scatter(x, y, label="Discrete Oscillogram")
    plt.plot(x, yhat, color="red", label="Continuous Estimate")
    plt.legend()
    plt.show()
    
    output_dir = r"plots/oscillo.png"
    plt.savefig(output_dir)
    
    

class Stream:
    def __init__(self):
        self.n = 0
        self.max_std = 0
        self.curr_time = 0
        self.curr_avg = 0
        self.curr_std = 0
        self.time_elapsed = 0
        self.times = []
        self.ppg = []
        self.force = []
        self.pct = 1.0
        self.prev_t = 1

    def update_avg_ppg(self, x):
        if self.n == 0:
            self.curr_avg = x
            self.n += 1
        self.curr_avg = self.curr_avg + (x - self.curr_avg) / self.n
        self.n += 1
        
    def update_std(self):
        self.curr_std = np.std(self.ppg[-13:])
        if self.curr_std > self.max_std:
            temp = self.curr_std
            self.max_std = temp
        self.pct = self.curr_std / self.max_std

# This function received serial data from the Arduino
def update_data():
    vout,ppg = receive_message(ser)
    timet = (time()-time1)

    return timet,float(vout),float(ppg)


stream = Stream()

# This function is called periodically from FuncAnimation
def animate(i):
    
    #for every 5 points plot 1,save data at 10hz
    if ((time() - time1) >= 60) or ((stream.pct < 0.20) and time()-time1 >= 50):
        close(ser)
        fig.savefig(r'plots/plot.png')
        plt.close(fig)
        save_data()
        return
    else:
        # Returns 3 floats
        time_t, vout_t, ppg_t = update_data()

        if (vout_t < 0.74):
            vout_t = (560*vout_t) - 281.9
        else:
            vout_t = (225.2*vout_t) - 32.1 

        if (vout_t < 0):
            vout_t = 0
        
        stream.times.append(time_t)
        stream.force.append(vout_t)
        stream.ppg.append(ppg_t)
        stream.update_avg_ppg(ppg_t)
        
        delta_t = time_t - stream.prev_t
        if delta_t > 1.33 and time_t > 20:
            stream.update_std()
            stream.prev_t = time_t
            print("Max Std: {}".format(stream.max_std))
            print("Curr Std: {}".format(stream.curr_std))
            print("Pct: {}".format(stream.pct))
        

        # Draw x and y lists
        #ax.plot(xs, ys, c='black')       
        ax1.plot(stream.times, stream.ppg, c='black')       
        #print(ppg_t)
        
        
        # Smooth the force data. Should be a 1sec moving average. We know that
        # we sample at 5samples/1sec, so we hack around this and do a moving
        # average over 5 samples.
        avg_ys = moving_avg(stream.force, n=5)
         
            
        # Draw x and y lists
        ax.plot(stream.times, stream.force, c='black')
        ax.plot(stream.times, avg_ys, c='darkorange')       

def save_data():
    file = open("data/forcedata.txt","w")   
    file.writelines('time,force,ppg,sys_bp,dia_bp\n')
    for i in range(len(xs)):
        w_string = str(xs[i])+','+str(ys[i])+','+str(y2[i])+'\n'
        file.writelines(w_string)

def main():
    # Set up plot to call animate() function periodically

    
    ani = animation.FuncAnimation(fig, animate,
                                  #fargs=(stream),
                                  interval=1)
    plt.show()
    
    output_dir = r"plots"    
    #ani.save(output_dir, writer="Pillow")
    #fig.savefig('last_frame.png')
    
    time = np.array(stream.times)
    ap = np.array(stream.force)
    ppg = np.array(stream.ppg)
    fit_oscillogram(time, ap, ppg)
    


#%%


# Create figure for plotting
fig = plt.figure(figsize=(8, 7), dpi=70)

ax = fig.add_subplot(2, 1, 1)
ax.plot([10, 60], [140, 440], c = 'blue')
ax.plot([10, 60], [60, 360], c = 'blue')
#ax.plot([0, 30], [150, 150], c = 'black')
ax.plot([10,10],[0, 360], '--', c='blue')
ax.text(5, 6, r'test', fontsize=10)
# Format plot
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('Visual Finger Actuation Guide')
plt.xlabel('Time (s)')
plt.ylabel('Applied  finger pressure (mmHg)')

plt.xlim([0,60])
plt.ylim([0,360])
#plt.ylim([0,370])

ax1 = fig.add_subplot(2, 1, 2)
plt.title('PPG Signal in time')
plt.xlabel('Time (s)')
plt.ylabel('Blood volume')
plt.xlim([0,60])
#plt.ylim([3200,3400])
#plt.ylim([28000,35000])

# Start lists with zero values
xs = []
ys = []
y2 = []

# Get first time value
time1 = time()

# Initialize communication with SingleTact Sensor
ser = serial.Serial("COM8",baudrate = 57600)
ser.flushInput()
ser.flushOutput()



if __name__ == '__main__':
    main()
