import serial # the PySerial library
import time   # for timing purposes
import struct


def setup(serial_name, baud_rate):
    ser = serial.Serial(serial_name, baudrate=baud_rate)
    return ser

def close(ser):
    ser.close()

def send_message(ser, message):
   if(message[-1] != '\n'):
       message = message + '\n'
   ser.write(message.encode('utf-8'))

"""
Receive a message from Serial and limit it to num_bytes (default of 50)
"""
def receive_message(ser, num_bytes=32):  #32?

    var = None
    force = None
    ppg = None
    lst2 = []
    while(var==None):
        sz = ser.inWaiting()
        #print(sz)
        #ser.flushInput()
        var = ser.readline(32).decode('utf-8')
        #print(var)
        lst = (var.split())[0].split(",")
        #print(lst)
        if (len(lst)) > 1:
            #print(lst[0])
            #print(lst[1])
           
            force = lst[0]
            ppg = lst[1]
        else:
            force = 0
            ppg = 0
        ser.flushInput()
        ser.flush()

        #print(force)
        #print(ppg)
        return force,ppg


def main():
    ser = setup("COM3", 57600)
    # time.sleep(3)
    # send_message(ser, "hello world\n")
    while True:
        #time.sleep(0.3)
        message = receive_message(ser,6)
        #print(message)
    close(ser)

"""
Main entrypoint for the application
"""
if __name__== "__main__":
   main()
