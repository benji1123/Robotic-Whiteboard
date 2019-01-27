# Short bit of code to list the serial ports
#import serial.tools.list_ports
#print([comport.device for comport in serial.tools.list_ports.comports()])

import serial
from time import sleep
ser = serial.Serial('/dev/cu.usbmodem14101', 9600)
sleep(3)                    # A delay is needed here for some reason

f = open("code.txt", "r")
for line in f:
    print(line[:-1])
    ser.write( (line[:-1]).encode())
    sleep(5)

# At the end, go home
ser.write( ("P1").encode())
sleep(1)
ser.write( ("X330.0 Y330.0").encode())
