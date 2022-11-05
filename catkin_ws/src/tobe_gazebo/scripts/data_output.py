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
from sensor_msgs.msg import Imu

# this script subscribes to the /tobe/imu topic, generates uprightness and push detection metrics, and publishes those data
class DataOutput:
    def __init__(self):
        self.rhome=np.matrix([[1,0,0],[0,1,0],[0,0,1]]) # matrix to convert quaternion-based rotation matrix to desired world frame
        self.rm=np.matrix([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix for transforming body frame values to world frame
        self.init=False
        
        self.z=Vector3() # initialize uprightness vector 'x'
        self.accel=Vector3() # initialize acceleration vector 'accel'

        rospy.loginfo("5 seconds until IMU data logging begins...")
        rospy.sleep(5)
        rospy.loginfo("Publishing to topics /lean and /push begins now.")   
        self._sub_imu=rospy.Subscriber('/tobe/imu',Imu,self._imu_output, queue_size=1)
        self.runrate = 50 # run publishing loop ('talker') at this rate in Hz
        self._publish=self.talker()
        
        
    def _imu_output(self,msg):
        quat = msg.orientation
        #omega = msg.angular_velocity
        accel_raw = msg.linear_acceleration
        
        rm = self.UpdateVerticalAxisFromQuaternion(quat)
        self.ComputeAccelerationVector(rm,accel_raw)
 
    def UpdateVerticalAxisFromQuaternion(self,quat):
        w=quat.w
        x=quat.x
        y=quat.y
        z=quat.z
        n=norm([w,x,y,z],2)
        if n > 0.99 and n <= 1:
            rm0=np.matrix([[2*(w*w+x*x)-1, 2*(x*y-w*z), 2*(x*z+w*y)],[2*(x*y+w*z), 2*(w*w+y*y)-1, 2*(y*z-w*x)],[2*(x*z-w*y), 2*(y*z+w*x), 2*(w*w+z*z)-1]])
            if self.init==False:
                self.rhome=rm0.transpose()
                self.init=True
            self.rm=np.matmul(self.rhome,rm0)
            z_axis=Vector3()
            z_axis.x=self.rm[0,2]
            z_axis.y=self.rm[1,2]
            z_axis.z=self.rm[2,2]
            self.z=z_axis
            
            return self.rm

    def ComputeAccelerationVector(self,rotation_matrix,accel_raw):
        ax=accel_raw.x
        ay=accel_raw.y
        az=accel_raw.z
        a0=np.array([[ax],[ay],[az]])
        a=np.matmul(rotation_matrix,a0)
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
            lean.publish(self.z)
            push.publish(self.accel)
            rate.sleep()       

if __name__ == "__main__":
    rospy.init_node("data processing")
    rospy.sleep(1)

    rospy.loginfo("Processing data")
    detect = DataOutput()

