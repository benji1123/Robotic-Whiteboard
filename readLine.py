# Short bit of code to list the serial ports
import serial.tools.list_ports
import math
print([comport.device for comport in serial.tools.list_ports.comports()])

import serial
from time import sleep
sleep(3)                    # A delay is needed here for some reason

def draw():
    ser = serial.Serial('COM3', 9600)
    f = open("code.txt", "r")
    print("working")
    last = None
    sleep(3)
    for line in f:
        now = list(map(lambda x: int(x[1:]), line[:-1].split(' ')))
        if len(now) == 2:
            if last != None:
                dist = math.hypot(now[0] - last[0], now[1] - last[1])
                print(dist)
                sleep(max(1.5, 0.2 * dist))
            last = now
        else:
            sleep(1.5)
        print(line[:-1])
        ser.write((line[:-1]).encode())

    #At the end, go home
    ser.write( ("P1").encode())
    sleep(1)
    ser.write( ("X330.0 Y330.0").encode())
