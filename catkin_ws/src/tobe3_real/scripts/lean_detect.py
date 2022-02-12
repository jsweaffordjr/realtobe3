#!/usr/bin/python3
import roslib
import rospy
import copy
from std_msgs.msg import Float64
import math
import numpy as np
from numpy.linalg import norm
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

# this script subscribes to the /quat and /linear topics, generates uprightness and push detection metrics, and publishes those data
class LeanDetect:
    def __init__(self):
        self.rhome=np.matrix([[1,0,0],[0,1,0],[0,0,1]]) # matrix to convert quaternion-based rotation matrix to desired world frame
        self.rm=np.matrix([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix for transforming body frame values to world frame
        self.init=False
        self.x=Vector3() # initialize uprightness vector 'x'
        self.accel=Vector3() # initialize acceleration vector 'accel'
        self.calib=False
        rospy.sleep(5)
        rospy.loginfo("Waiting for IMU Calibration...")
        self.sub_cal=rospy.Subscriber('/calibration',Vector3,self._calibrate, queue_size=1)
        while self.calib == False:
            rospy.sleep(0.1)
        rospy.loginfo("10 seconds until IMU data logging begins...")
        rospy.sleep(10) 
        rospy.loginfo("Publishing to topics /lean and /push begins now.")   
        self._sub_quat=rospy.Subscriber('/quat',Quaternion,self._upright, queue_size=1)
        self._sub_acc=rospy.Subscriber('/linear',Vector3,self._push, queue_size=1)
        self.runrate = 50 # run publishing loop ('talker') at this rate in Hz
        self._publish=self.talker()
    
    def _calibrate(self,msg):
        x=msg.x
        y=msg.y
        z=msg.z
        if x > 2:
            if y > 2: 
                if z > 2:
                    self.calib=True
        
    def _upright(self,msg):
        w=msg.w
        x=msg.x
        y=msg.y
        z=msg.z
        n=norm([w,x,y,z],2)
        if n > 0.99 and n <= 1:
            rm0=np.matrix([[2*(w*w+x*x)-1, 2*(x*y-w*z), 2*(x*z+w*y)],[2*(x*y+w*z), 2*(w*w+y*y)-1, 2*(y*z-w*x)],[2*(x*z-w*y), 2*(y*z+w*x), 2*(w*w+z*z)-1]])
            if self.init==False:
                self.rhome=rm0.transpose()
                self.init=True
            self.rm=np.matmul(self.rhome,rm0)
            x_axis=Vector3()
            x_axis.x=self.rm[0,0]
            x_axis.y=self.rm[1,0]
            x_axis.z=self.rm[2,0]
            self.x=x_axis

    def _push(self,msg):
        ax=msg.x
        ay=msg.y
        az=msg.z
        a0=np.array([[ax],[ay],[az]])
        a=np.matmul(self.rm,a0)
        a1=Vector3()
        a1.x=a[0]
        a1.y=a[1]
        a1.z=a[2]
        self.accel=a1
    
    def talker(self):
        lean = rospy.Publisher('lean', Vector3, queue_size=1)
        push = rospy.Publisher('push', Vector3, queue_size=1)
        rate=rospy.Rate(self.runrate)
        rospy.sleep(2.0)
        while not rospy.is_shutdown():
            lean.publish(self.x)
            push.publish(self.accel)
            rate.sleep()       

if __name__ == "__main__":
    rospy.init_node("stand_straight")
    rospy.sleep(1)

    rospy.loginfo("Instantiating Lean Detection")
    detect = LeanDetect()

