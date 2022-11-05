import random
from threading import Thread
import math
import rospy
import time
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3

class Tobe:

    def __init__(self,ns="/tobe/"):
        self.ns=ns
        self.joints=None
        self.angles={}
        
        self._sub_joints=rospy.Subscriber(ns+"joint_states",JointState,self._cb_joints,queue_size=1)
        rospy.loginfo("+Waiting for joints to be populated...")
        while not rospy.is_shutdown():
            if self.joints is not None: break
            rospy.sleep(0.1)            
        rospy.loginfo(" -Joints populated: "+str(len(self.joints)))
          
        rospy.loginfo("+Creating joint command publishers...")
        self._pub_joints={}
        for j in self.joints:
            p=rospy.Publisher(self.ns+j+"_position_controller/command",Float64, queue_size=10)
            self._pub_joints[j]=p
            rospy.loginfo(" -Found: "+j)
        
        rospy.sleep(1)


    def _cb_joints(self,msg):
        if self.joints is None:
            self.joints=msg.name
        
        for k in range(len(self.joints)):   
            self.angles[self.joints[k]]=msg.position[k]            
        
    def command_all_motors(self,angs):
        # this sends motor commands to all joints by publishing commanded joint angles to all 18 topics
        self._pub_joints["r_shoulder_swing_joint"].publish(angs[0])
        self._pub_joints["l_shoulder_swing_joint"].publish(angs[1])
        self._pub_joints["r_shoulder_lateral_joint"].publish(angs[2])
        self._pub_joints["l_shoulder_lateral_joint"].publish(angs[3])
        self._pub_joints["r_elbow_joint"].publish(angs[4])
        self._pub_joints["l_elbow_joint"].publish(angs[5])
        self._pub_joints["r_hip_twist_joint"].publish(angs[6])
        self._pub_joints["l_hip_twist_joint"].publish(angs[7])
        self._pub_joints["r_hip_lateral_joint"].publish(angs[8])
        self._pub_joints["l_hip_lateral_joint"].publish(angs[9])
        self._pub_joints["r_hip_swing_joint"].publish(angs[10])
        self._pub_joints["l_hip_swing_joint"].publish(angs[11])
        self._pub_joints["r_knee_joint"].publish(angs[12])
        self._pub_joints["l_knee_joint"].publish(angs[13])
        self._pub_joints["r_ankle_swing_joint"].publish(angs[14])
        self._pub_joints["l_ankle_swing_joint"].publish(angs[15])
        self._pub_joints["r_ankle_lateral_joint"].publish(angs[16])
        self._pub_joints["l_ankle_lateral_joint"].publish(angs[17])

    def command_sag_motors(self,angs):
        # this function sends commands to the motors for sagittal shoulders (ID:1,2), hips (11,12), knees (13,14) and ankles (15,16) by publishing the commanded angles to the corresponding publisher topics:
        # NOTE: 'cmds' is just a 8-element array of the 10-bit values to the joints in that order
        self._pub_joints["r_shoulder_swing_joint"].publish(angs[0])
        self._pub_joints["l_shoulder_swing_joint"].publish(angs[1])
        self._pub_joints["r_hip_swing_joint"].publish(angs[2])
        self._pub_joints["l_hip_swing_joint"].publish(angs[3])
        self._pub_joints["r_knee_joint"].publish(angs[4])
        self._pub_joints["l_knee_joint"].publish(angs[5]) 
        self._pub_joints["r_ankle_swing_joint"].publish(angs[6])
        self._pub_joints["l_ankle_swing_joint"].publish(angs[7]) 
    
    def read_sag_angles(self):
        # this function reads the motor positions for the sagittal joints
        p1=self.angles["r_shoulder_swing_joint"]
        p2=self.angles["l_shoulder_swing_joint"]
        p3=self.angles["r_hip_swing_joint"]
        p4=self.angles["l_hip_swing_joint"]
        p5=self.angles["r_knee_joint"]
        p6=self.angles["l_knee_joint"]
        p7=self.angles["r_ankle_swing_joint"]
        p8=self.angles["l_ankle_swing_joint"]
        p=[p1,p2,p3,p4,p5,p6,p7,p8]
        return p

    def read_all_angles(self):
        # this function reads all motor positions
        p1=self._pub_joints["r_shoulder_swing_joint"]
        p2=self._pub_joints["l_shoulder_swing_joint"]
        p3=self._pub_joints["r_shoulder_lateral_joint"]
        p4=self._pub_joints["l_shoulder_lateral_joint"]
        p5=self._pub_joints["r_elbow_joint"]
        p6=self._pub_joints["l_elbow_joint"]
        p7=self._pub_joints["r_hip_twist_joint"]
        p8=self._pub_joints["l_hip_twist_joint"]
        p9=self._pub_joints["r_hip_lateral_joint"]
        p10=self._pub_joints["l_hip_lateral_joint"]
        p11=self._pub_joints["r_hip_swing_joint"]
        p12=self._pub_joints["l_hip_swing_joint"]
        p13=self._pub_joints["r_knee_joint"]
        p14=self._pub_joints["l_knee_joint"]
        p15=self._pub_joints["r_ankle_swing_joint"]
        p16=self._pub_joints["l_ankle_swing_joint"]
        p17=self._pub_joints["r_ankle_lateral_joint"]
        p18=self._pub_joints["l_ankle_lateral_joint"]
        p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]
        return p          

    def convert_motor_positions_to_angles(self,ids,cmds):
        # this function converts an array of angle values (in radians) to the corresponding 10-bit motor values,
        # assuming that the 'ids' array contains matching ID numbers for the angle values of 'angles' array
        
        b=[60,240,60,240,150,150,150,150,150,150,150,150,150,150,150,150,150,150] # motor offsets
        c=[1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1] # motor polarities
        
        angs=np.zeros(len(ids)) # initialize cmds array
        for j in range(len(ids)): # get 10-bit motor values:
            num=ids[j] # get motor ID number from array
            cmd=cmds[j] # get motor position as 10-bit value
            angs[j]=(((300/1023)*cmd)-b[num-1])*c[num-1]*(math.pi/180) # convert from 10-bit, subtract offset, apply polarity, go to degrees
        return angs 
