import matplotlib.pyplot as plt
import matplotlib.animation as animation
from arduinoIO import setup, close, receive_message
from time import time
from time import sleep
import serial 

# Create figure for plotting
fig = plt.figure(figsize=(8, 7), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.plot([0, 20], [60, 150], c = 'blue')
ax.plot([0, 30], [0, 150], c = 'blue')
ax.plot([10,10],[0, 160], '--', c='blue')
ax.text(5, 6, r'test', fontsize=10)

# Format plot force
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('Visual Finger Actuation Guide')
plt.xlabel('Time (s)')
plt.ylabel('Applied  finger pressure (mmHg')

plt.xlim([0,40])
plt.ylim([0,160])

# Format plot ppg
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('Visual Finger Actuation Guide')
plt.xlabel('Time (s)')
plt.ylabel('Applied  finger pressure (mmHg)')

plt.xlim([0,40])
plt.ylim([0,160])
#plt.ylim([0,370])

ax1 = fig.add_subplot(2, 1, 2)
plt.title('PPG Signal in time')
plt.xlabel('Time (s)')
plt.ylabel('Blood volume')
plt.xlim([0,40])
plt.ylim([2400,2900])

# Start lists with zero values
xs = [0]
ys = [0]
y2 = [0]

# Get first time value
time1 = time()

# Initialize communication with SingleTact Sensor
ser = setup("COM6", 57600)
ser.flushInput()
ser.flushOutput()

# This function received serial data from the Arduino
def update_data():
    vout,ppg = receive_message(ser)
    timet = (time()-time1)

    return timet,float(vout),float(ppg)

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    #for every 5 points plot 1,save data at 10hz
    if ((time() - time1) >= 40):
        close(ser)
        fig.savefig('plot.png')
        plt.close(fig)
        save_data()
        return
    else:
        # Add x and y to lists
        col1,col2,col3 = update_data()

        if (col2 < 0.74):
            col2 = 560*col2-296.8
        else:
            col2 = 230*col2-30  

        if (col2 < 0):
            col2 = 0

        xs.append(col1)  #time
        ys.append(col2-100)  #force, added scaling for allow users to put more force to ease the plot
        #print(col2-125)
        y2.append(col3*10)  #ppg, added scaling for bigger resolution
        
        # Draw x and y lists
        ax.plot(xs, ys, c='black')       
        ax1.plot(xs, y2, c='black')           

def save_data():
    file = open("forceppgdata.txt","w")
    for i in range(len(xs)):
        file.writelines(str(xs[i])+','+str(ys[i])+'\n')

def main():
    # Set up plot to call animate() function periodically
    ta = time()
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1)
    plt.show()
    tb = time()

if __name__ == '__main__':
    main()
