#!/usr/bin/python3
import roslib
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64
import serial
import math
import numpy as np
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

# this script connects to USB port for reading BNO055 IMU sensor via Arduino

ser = serial.Serial('/dev/ttyUSB0', 115200)
runrate = 50 # run loop at this rate in Hz

def talker():
 linear = rospy.Publisher('linear', Vector3, queue_size=1)
 quat = rospy.Publisher('quat', Quaternion, queue_size=1)
 calib = rospy.Publisher('calibration', Vector3, queue_size=1)
 rospy.init_node('talker', anonymous=True)
 rate=rospy.Rate(runrate)
 rospy.sleep(2.0)
 while not rospy.is_shutdown():
   # read five lines from serial port:
   data1= ser.readline() # read single line from serial
   data2= ser.readline() # read single line from serial
   data3= ser.readline() # read single line from serial
   data4= ser.readline() # read single line from serial
   
   # convert serial communication lines to strings
   msg1=str(data1)
   msg2=str(data2)
   msg3=str(data3)
   msg4=str(data4)
   msgs=[msg1,msg2,msg3,msg4] # compile strings into list
   
   # determine strings to search for to determine appropriate publisher
   cstr="Calib"
   qstr="Quat"
   lstr="Linear"
   
   # create blank messages
   cmsg=Vector3()
   qmsg=Quaternion()
   lmsg=Vector3()

   # assign each message to the appropriate publisher
   try:
      for j in msgs:
         if j.find(cstr)>0:
            gloc=j.find('Gyro=')
            aloc=j.find('Accel=')
            mloc=j.find('Mag=')
            try:
               cmsg.x=int(j[gloc+5:gloc+6])
               cmsg.y=int(j[aloc+6:aloc+7])
               cmsg.z=int(j[mloc+4:mloc+5])
               calib.publish(cmsg)
            except:
               cmsg.x=-1
               cmsg.y=-1
               cmsg.z=-1          
         elif j.find(lstr)>0:
            xloc=j.find('x=')
            xstr=j[xloc:xloc+10]
            xend=xstr.find(' |')
            yloc=j.find('y=')
            ystr=j[yloc:yloc+10]
            yend=ystr.find(' |')
            zloc=j.find('z=')
            zstr=j[zloc:zloc+10]
            zend=zstr.find('\\')
            lmsgx=xstr[3:xend]
            lmsgy=ystr[3:yend]
            lmsgz=zstr[3:zend]
            lmsg1="%s %s %s"%(lmsgx,lmsgy,lmsgz)
            lmsg2=lmsg1.strip("\"")
            chunks=lmsg2.split(' ')
            try:
               lmsg.x=float(chunks[0])
               lmsg.y=float(chunks[1])
               lmsg.z=float(chunks[2])
               linear.publish(lmsg)
            except:
               lmsg.x=-1
               lmsg.y=-1
               lmsg.z=-1
         elif j.find(qstr)>0:
            wloc=j.find('w=')
            wstr=j[wloc:wloc+10]
            wend=wstr.find(' |')
            xloc=j.find('x=')
            xstr=j[xloc:xloc+10]
            xend=xstr.find(' |')
            yloc=j.find('y=')
            ystr=j[yloc:yloc+10]
            yend=ystr.find(' |')
            zloc=j.find('z=')
            zstr=j[zloc:zloc+10]
            zend=zstr.find('\\')
            qmsgw=wstr[3:wend]
            qmsgx=xstr[3:xend]
            qmsgy=ystr[3:yend]
            qmsgz=zstr[3:zend]
            qmsg1="%s %s %s %s"%(qmsgw,qmsgx,qmsgy,qmsgz)
            qmsg2=qmsg1.strip("\"")
            chunks=qmsg2.split(' ')
            try:
               q1=float(chunks[0])
               q2=float(chunks[1])
               q3=float(chunks[2])
               q4=float(chunks[3])
               if ((abs(q1) < 1) and (abs(q2) < 1) and (abs(q3) < 1) and (abs(q4) < 1)):
                   qmsg.w=q1
                   qmsg.x=q2
                   qmsg.y=q3
                   qmsg.z=q4
                   quat.publish(qmsg)
            except:
               qmsg.w=-1
               qmsg.x=-1
               qmsg.y=-1
               qmsg.z=-1
   except:
      cmsg.x=0
      cmsg.y=0
      cmsg.z=0
      lmsg.x=-1
      lmsg.y=-1
      lmsg.z=-1
      qmsg.w=-1
      qmsg.x=-1
      qmsg.y=-1
      qmsg.z=-1
      
   rate.sleep()


if __name__ == '__main__':
  try:
    talker()
  except rospy.ROSInterruptException:
    pass
