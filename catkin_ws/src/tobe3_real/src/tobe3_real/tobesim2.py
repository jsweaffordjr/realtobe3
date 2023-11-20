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
        self.ang_names={}
        self.torques={}
        self.cmd_angs={}
        
        self._sub_joints=rospy.Subscriber(ns+"joint_states",JointState,self._cb_joints,queue_size=1)
        rospy.loginfo("+Waiting for joints to be populated...")
        while not rospy.is_shutdown():
            if self.joints is not None: break
            rospy.sleep(0.1)            
        rospy.loginfo(" -Joints populated: "+str(len(self.joints)))
          
        rospy.loginfo("+Creating joint command publishers...")
        self._pub_joints={}
        #cmds_init = [0.39,-0.39,-0.35,-0.35,0.52,-0.52,0,0,0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18]
        cmds_init = [-0.18,-1.11,-0.52,-0.16,0.5,0,-1.74,-0.35,-0.39,0.18,1.11,0.52,0.16,-0.5,0,1.74,-0.35,0.39]
        for i,j in enumerate(self.joints):
            p=rospy.Publisher(self.ns+j+"_torque_controller/command",Float64, queue_size=10)
            self._pub_joints[j]=p
            self.cmd_angs[j]=cmds_init[i]
            rospy.loginfo(" -Found: "+j)

        rospy.sleep(1) 
        
                         
    def _cb_joints(self,msg):
        if self.joints is None:
            self.joints=msg.name        
        
        for k in range(len(self.joints)):   
            self.angles[self.joints[k]]=msg.position[k]  
            self.torques[self.joints[k]]=msg.effort[k] 
                                      
    def compute_torque(self, joint_ang, cmd):
        # this computes joint torque based on current joint angle and command
        z = np.round(joint_ang, decimals=3)
        y = np.round(cmd, decimals=3)
        error = np.round(z - y, 3)
        #print(f"Error: {errors}")
        max_torque = 5.0
        torque = 0.0
        # set torque control       
        if np.abs(error) > np.pi/18.0: # abs(error) > 10 deg
            if error > 0: # positive error > 10 deg
                torque = -max_torque # N*m
            else: # negative error < -10 deg
                torque = max_torque # N*m
        elif np.abs(error) > np.pi/600.0: # 0.3 deg < abs(error) <= 10 deg
            if error > 0: # positive error, 0.3 deg < error <= 10 deg
                torque = -max_torque + (0.97*max_torque / (9.7 * (np.pi/180.0)))*((np.pi/18.0) - error)
            else: # negative error, -10 deg <= error < -0.3 deg
                torque = max_torque - (0.97*max_torque / (9.7 * (np.pi/180.0)))*((np.pi/18.0) + error)    
        # otherwise, use zero torque for -0.3 deg <= error <= 0.3 deg
            
        # round values, apply torque control
        torque = np.round(torque, 2)
        
        return torque
                 
        
    def command_all_motors(self,cmd_angs):    
        # this sends motor commands to all joints by publishing joint commands to all 18 topics
        self.cmd_angs["r_shoulder_swing_joint"]=cmd_angs[0]
        self.cmd_angs["l_shoulder_swing_joint"]=cmd_angs[1]
        self.cmd_angs["r_shoulder_lateral_joint"]=cmd_angs[2]
        self.cmd_angs["l_shoulder_lateral_joint"]=cmd_angs[3]
        self.cmd_angs["r_elbow_joint"]=cmd_angs[4]
        self.cmd_angs["l_elbow_joint"]=cmd_angs[5]
        self.cmd_angs["r_hip_twist_joint"]=cmd_angs[6]
        self.cmd_angs["l_hip_twist_joint"]=cmd_angs[7]
        self.cmd_angs["r_hip_lateral_joint"]=cmd_angs[8]
        self.cmd_angs["l_hip_lateral_joint"]=cmd_angs[9]
        self.cmd_angs["r_hip_swing_joint"]=cmd_angs[10]
        self.cmd_angs["l_hip_swing_joint"]=cmd_angs[11]
        self.cmd_angs["r_knee_joint"]=cmd_angs[12]
        self.cmd_angs["l_knee_joint"]=cmd_angs[13]
        self.cmd_angs["r_ankle_swing_joint"]=cmd_angs[14]
        self.cmd_angs["l_ankle_swing_joint"]=cmd_angs[15]
        self.cmd_angs["r_ankle_lateral_joint"]=cmd_angs[16]
        self.cmd_angs["l_ankle_lateral_joint"]=cmd_angs[17]
        
        for k in range(len(self.joints)):
            cmd = self.compute_torque(self.angles[self.joints[k]], self.cmd_angs[self.joints[k]])
            self._pub_joints[self.joints[k]].publish(cmd)
        
    
    def command_leg_motors(self,cmd_angs):
        # this function controls the motors for the 10 lower-leg angles by publishing the commands to the corresponding publisher topics:
        self.cmd_angs["r_hip_lateral_joint"]=cmd_angs[0]
        self.cmd_angs["l_hip_lateral_joint"]=cmd_angs[1]
        self.cmd_angs["r_hip_swing_joint"]=cmd_angs[2]
        self.cmd_angs["l_hip_swing_joint"]=cmd_angs[3]
        self.cmd_angs["r_knee_joint"]=cmd_angs[4]
        self.cmd_angs["l_knee_joint"]=cmd_angs[5]
        self.cmd_angs["r_ankle_swing_joint"]=cmd_angs[6]
        self.cmd_angs["l_ankle_swing_joint"]=cmd_angs[7]
        self.cmd_angs["r_ankle_lateral_joint"]=cmd_angs[8]
        self.cmd_angs["l_ankle_lateral_joint"]=cmd_angs[9]
    
    def command_sag_motors(self,cmds):
        # this function controls the motors for sagittal shoulders (ID:1,2), hips (11,12), knees (13,14) and ankles (15,16) by publishing the commands to the corresponding publisher topics:
        
        self.cmd_angs["r_shoulder_swing_joint"]=cmd_angs[0]
        self.cmd_angs["l_shoulder_swing_joint"]=cmd_angs[1]
        self.cmd_angs["r_hip_swing_joint"]=cmd_angs[2]
        self.cmd_angs["l_hip_swing_joint"]=cmd_angs[3]
        self.cmd_angs["r_knee_joint"]=cmd_angs[4]
        self.cmd_angs["l_knee_joint"]=cmd_angs[5]
        self.cmd_angs["r_ankle_swing_joint"]=cmd_angs[6]
        self.cmd_angs["l_ankle_swing_joint"]=cmd_angs[7] 
    
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
    
    def read_leg_angles(self):
        # this function reads the motor positions for the ten lower-leg joints
        p1=self.angles["r_hip_lateral_joint"]
        p2=self.angles["l_hip_lateral_joint"]
        p3=self.angles["r_hip_swing_joint"]
        p4=self.angles["l_hip_swing_joint"]
        p5=self.angles["r_knee_joint"]
        p6=self.angles["l_knee_joint"]
        p7=self.angles["r_ankle_swing_joint"]
        p8=self.angles["l_ankle_swing_joint"]
        p9=self.angles["r_ankle_lateral_joint"]
        p10=self.angles["l_ankle_lateral_joint"]
        p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
        return p
        
    def read_all_angles(self):
        # this function reads all motor positions
        p1=self.angles["r_shoulder_swing_joint"]
        p2=self.angles["l_shoulder_swing_joint"]
        p3=self.angles["r_shoulder_lateral_joint"]
        p4=self.angles["l_shoulder_lateral_joint"]
        p5=self.angles["r_elbow_joint"]
        p6=self.angles["l_elbow_joint"]
        p7=self.angles["r_hip_twist_joint"]
        p8=self.angles["l_hip_twist_joint"]
        p9=self.angles["r_hip_lateral_joint"]
        p10=self.angles["l_hip_lateral_joint"]
        p11=self.angles["r_hip_swing_joint"]
        p12=self.angles["l_hip_swing_joint"]
        p13=self.angles["r_knee_joint"]
        p14=self.angles["l_knee_joint"]
        p15=self.angles["r_ankle_swing_joint"]
        p16=self.angles["l_ankle_swing_joint"]
        p17=self.angles["r_ankle_lateral_joint"]
        p18=self.angles["l_ankle_lateral_joint"]
        p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]
        return p          


