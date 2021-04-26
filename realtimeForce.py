import matplotlib.pyplot as plt
import matplotlib.animation as animation
from arduinoIO import setup, close, receive_message
from time import time
from time import sleep
import serial 

# Create figure for plotting
fig = plt.figure(figsize=(8, 7), dpi=70)
ax = fig.add_subplot(1, 1, 1)
ax.plot([0, 20], [60, 150], c = 'blue')
ax.plot([0, 30], [0, 150], c = 'blue')

# Format plot
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('Visual Finger Actuation Guide')
plt.xlabel('Time (s)')
plt.ylabel('Applied  finger pressure (mmHg')

plt.xlim([0,30])
plt.ylim([0,170])

# Start lists with zero values
xs = [0]
ys = [0]

# Get first time value
time1 = time()

# Initialize communication with SingleTact Sensor
ser = serial.Serial("COM3",baudrate = 115200)
ser.flushInput()
ser.flushOutput()

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    #for every 5 points plot 1,save data at 10hz
    if ((time() - time1) >= 30):
        close(ser)
        fig.savefig('plot.png')
        plt.close(fig)
        save_data()
        return
    else:
        forcef = None
        while(forcef == None):
            sz = ser.inWaiting()
            print(sz)
            ser.flushInput()
            forcef = ser.readline(32).decode()    
            ser.flushInput()
            ser.flush()

        xs.append(time()-time1)  #time
if (col2 < 0.74):
            col2 = 560*col2-296.8
        else:
            col2 = 230*col2-30  

        ys.append(float(forcef))  #force

        # Draw x and y lists
        ax.plot(xs, ys, c='black')       

def save_data():
    file = open("forcedata.txt","w")
    for i in range(len(xs)):
        file.writelines(str(xs[i])+','+str(ys[i])+'\n')

def main():
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1)
    plt.show()

if __name__ == '__main__':
    main()