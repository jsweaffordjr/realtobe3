import random
from threading import Thread
import math
import rospy
import time
import numpy as np
from dynio import *
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3

class Tobe:

    def __init__(self,ns="/realtobe/"):
        self.ns=ns
        # array of TOBE joint names in Robotis order
        self.joints=["r_shoulder_sagittal","l_shoulder_sagittal","r_shoulder_frontal","l_shoulder_frontal","r_elbow","l_elbow",
        "r_hip_swivel","l_hip_swivel","r_hip_frontal","l_hip_frontal","r_hip_sagittal","l_hip_sagittal","r_knee","l_knee",
        "r_ankle_sagittal","l_ankle_sagittal","r_ankle_frontal","l_ankle_frontal"]
        
        # initialize tobe      
        tobe = dxl.DynamixelIO('/dev/ttyUSB0',1000000) # port being used, e.g., "/dev/ttyUSB0" and baud rate (e.g., 1000000)
        
        # set up motor connections
        self.motor01 = tobe.new_ax12(1)        # motor01: r_shoulder_sagittal
        self.motor02 = tobe.new_ax12(2)        # motor02: l_shoulder_sagittal
        self.motor03 = tobe.new_ax12(3)        # motor03: r_shoulder_frontal
        self.motor04 = tobe.new_ax12(4)        # motor04: l_shoulder_frontal
        self.motor05 = tobe.new_ax12(5)        # motor05: r_elbow
        self.motor06 = tobe.new_ax12(6)        # motor06: l_elbow
        self.motor07 = tobe.new_ax12(7)        # motor07: r_hip_swivel
        self.motor08 = tobe.new_ax12(8)        # motor08: l_hip_swivel
        self.motor09 = tobe.new_ax12(9)        # motor09: r_hip_frontal
        self.motor10 = tobe.new_ax12(10)       # motor10: l_hip_frontal
        self.motor11 = tobe.new_ax12(11)       # motor11: r_hip_sagittal
        self.motor12 = tobe.new_ax12(12)       # motor12: l_hip_sagittal
        self.motor13 = tobe.new_ax12(13)       # motor13: r_knee
        self.motor14 = tobe.new_ax12(14)       # motor14: l_knee
        self.motor15 = tobe.new_ax12(15)       # motor15: r_ankle_sagittal
        self.motor16 = tobe.new_ax12(16)       # motor16: l_ankle_sagittal
        self.motor17 = tobe.new_ax12(17)       # motor17: r_ankle_frontal
        self.motor18 = tobe.new_ax12(18)       # motor18: l_ankle_frontal     
        rospy.loginfo("Motor connections established.")
        
        # turn on motors:
        self.turn_on_motors()
        
        # create joint command publishers
        rospy.loginfo("+Creating joint publishers...")
        self._pub_joints={}
        self._pub_joint_cmds={}
        self._pub_joint_vels={}
        for j in self.joints:
            p=rospy.Publisher(self.ns+j+"/command",Float64, queue_size=10)
            q=rospy.Publisher(self.ns+j+"/angle",Float64, queue_size=10)
            w=rospy.Publisher(self.ns+j+"/vels",Float64, queue_size=10)
            self._pub_joint_cmds[j]=p
            self._pub_joints[j]=q
            self._pub_joint_vels[j]=w
            rospy.loginfo(" -Found: "+j)
            
    def turn_on_motors(self):
        self.motor01.torque_enable()
        self.motor02.torque_enable()
        self.motor03.torque_enable()
        self.motor04.torque_enable()
        self.motor05.torque_enable()
        self.motor06.torque_enable()
        self.motor07.torque_enable()
        self.motor08.torque_enable()
        self.motor09.torque_enable()
        self.motor10.torque_enable()
        self.motor11.torque_enable()
        self.motor12.torque_enable()
        self.motor13.torque_enable()
        self.motor14.torque_enable()
        self.motor15.torque_enable()
        self.motor16.torque_enable()
        self.motor17.torque_enable()
        self.motor18.torque_enable()
        
    def command_all_motors(self,cmds):
        # this function sends motor commands to all 18 joints
        self.motor01.set_position(int(cmds[0]))
        self.motor02.set_position(int(cmds[1]))
        self.motor03.set_position(int(cmds[2]))
        self.motor04.set_position(int(cmds[3]))
        self.motor05.set_position(int(cmds[4]))
        self.motor06.set_position(int(cmds[5]))
        self.motor07.set_position(int(cmds[6]))
        self.motor08.set_position(int(cmds[7]))
        self.motor09.set_position(int(cmds[8]))
        self.motor10.set_position(int(cmds[9]))
        self.motor11.set_position(int(cmds[10]))
        self.motor12.set_position(int(cmds[11]))
        self.motor13.set_position(int(cmds[12]))
        self.motor14.set_position(int(cmds[13]))
        self.motor15.set_position(int(cmds[14]))
        self.motor16.set_position(int(cmds[15]))
        self.motor17.set_position(int(cmds[16]))
        self.motor18.set_position(int(cmds[17]))
            
    def publish_all_motor_commands(self,angs):
        # this function publishes commanded joint angles to all 18 publisher topics
        self._pub_joints["r_shoulder_sagittal"].publish(angs[0])
        self._pub_joints["l_shoulder_sagittal"].publish(angs[1])
        self._pub_joints["r_shoulder_frontal"].publish(angs[2])
        self._pub_joints["l_shoulder_frontal"].publish(angs[3])
        self._pub_joints["r_elbow"].publish(angs[4])
        self._pub_joints["l_elbow"].publish(angs[5])
        self._pub_joints["r_hip_swivel"].publish(angs[6])
        self._pub_joints["l_hip_swivel"].publish(angs[7])
        self._pub_joints["r_hip_frontal"].publish(angs[8])
        self._pub_joints["l_hip_frontal"].publish(angs[9])
        self._pub_joints["r_hip_sagittal"].publish(angs[10])
        self._pub_joints["l_hip_sagittal"].publish(angs[11])
        self._pub_joints["r_knee"].publish(angs[12])
        self._pub_joints["l_knee"].publish(angs[13])
        self._pub_joints["r_ankle_sagittal"].publish(angs[14])
        self._pub_joints["l_ankle_sagittal"].publish(angs[15])
        self._pub_joints["r_ankle_frontal"].publish(angs[16])
        self._pub_joints["l_ankle_frontal"].publish(angs[17])

    def command_arm_motors(self,cmds):
        # this function sends commands to the arm motors (1-6)
        # 'cmds' is just a 6-element array of the 10-bit values to the joints in that order
        self.motor01.set_position(int(cmds[0]))
        self.motor02.set_position(int(cmds[1]))
        self.motor03.set_position(int(cmds[2]))
        self.motor04.set_position(int(cmds[3]))
        self.motor05.set_position(int(cmds[4]))
        self.motor06.set_position(int(cmds[5]))
    
    def command_leg_motors(self,cmds):
        # this function sends commands to the leg motors (hips (9,10,11,12), knees (13,14) and ankles (15,16,17,18))
        # 'cmds' is just a 10-element array of the 10-bit values to the joints in that order
        self.motor09.set_position(int(cmds[0]))
        self.motor10.set_position(int(cmds[1]))
        self.motor11.set_position(int(cmds[2]))
        self.motor12.set_position(int(cmds[3]))
        self.motor13.set_position(int(cmds[4]))
        self.motor14.set_position(int(cmds[5]))
        self.motor15.set_position(int(cmds[6]))
        self.motor16.set_position(int(cmds[7]))
        self.motor17.set_position(int(cmds[8]))
        self.motor18.set_position(int(cmds[9]))
        
    def command_sag_motors(self,cmds):
        # this function sends commands to the motors for sagittal shoulders (ID:1,2), hips (11,12), knees (13,14) and ankles (15,16)
        # 'cmds' is just a 8-element array of the 10-bit values to the joints in that order
        self.motor01.set_position(int(cmds[0]))
        self.motor02.set_position(int(cmds[1]))
        self.motor11.set_position(int(cmds[2]))
        self.motor12.set_position(int(cmds[3]))
        self.motor13.set_position(int(cmds[4]))
        self.motor14.set_position(int(cmds[5]))
        self.motor15.set_position(int(cmds[6]))
        self.motor16.set_position(int(cmds[7]))
    
    def read_all_motor_positions(self):
        # this function reads the motor positions for all joints
        p01=self.motor01.get_position()
        p02=self.motor02.get_position()
        p03=self.motor03.get_position()
        p04=self.motor04.get_position()
        p05=self.motor05.get_position()
        p06=self.motor06.get_position()
        p07=self.motor07.get_position()
        p08=self.motor08.get_position()
        p09=self.motor09.get_position()
        p10=self.motor10.get_position()
        p11=self.motor11.get_position()
        p12=self.motor12.get_position()
        p13=self.motor13.get_position()
        p14=self.motor14.get_position()
        p15=self.motor15.get_position()
        p16=self.motor16.get_position()
        p17=self.motor17.get_position()
        p18=self.motor18.get_position()
        p=[p01,p02,p03,p04,p05,p06,p07,p08,p09,p10,p11,p12,p13,p14,p15,p16,p17,p18]
        return p
    
    def read_leg_motor_positions(self):
        # this function reads the motor positions for the leg joints
        p1=self.motor09.get_position()
        p2=self.motor10.get_position()
        p3=self.motor11.get_position()
        p4=self.motor12.get_position()
        p5=self.motor13.get_position()
        p6=self.motor14.get_position()
        p7=self.motor15.get_position()
        p8=self.motor16.get_position()
        p9=self.motor17.get_position()
        p0=self.motor18.get_position()
        p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p0]
        return p
    
    def read_sag_motor_positions(self):
        # this function reads the motor positions for the sagittal joints
        p1=self.motor01.get_position()
        p2=self.motor02.get_position()
        p3=self.motor11.get_position()
        p4=self.motor12.get_position()
        p5=self.motor13.get_position()
        p6=self.motor14.get_position()
        p7=self.motor15.get_position()
        p8=self.motor16.get_position()
        p=[p1,p2,p3,p4,p5,p6,p7,p8]
        return p
    
    def read_arm_motor_positions(self):
        # this function reads the motor positions for the arm joints
        p1=self.motor01.get_position()
        p2=self.motor02.get_position()
        p3=self.motor03.get_position()
        p4=self.motor04.get_position()
        p5=self.motor05.get_position()
        p6=self.motor06.get_position()
        p=[p1,p2,p3,p4,p5,p6]
        return p
        
    def read_right_arm_motor_positions(self):
        # this function reads the motor positions for the arm joints
        p1=self.motor01.get_position() # sag. shoulder
        p2=self.motor03.get_position() # frontal shoulder
        p3=self.motor05.get_position() # elbow
        p=[p1,p2,p3]
        return p
        
    def read_right_sag_arm_motor_positions(self):
        # this function reads the motor positions for the arm joints
        p1=self.motor01.get_position()
        p2=self.motor05.get_position()
        p=[p1,p2]
        return p
    
    def publish_arm_cmds(self,angs):
        # the function publishes the commanded angles to the corresponding publisher topics:
        self._pub_joint_cmds["r_shoulder_sagittal"].publish(angs[0])
        self._pub_joint_cmds["l_shoulder_sagittal"].publish(angs[1])
        self._pub_joint_cmds["r_shoulder_frontal"].publish(angs[2])
        self._pub_joint_cmds["l_shoulder_frontal"].publish(angs[3])
        self._pub_joint_cmds["r_elbow"].publish(angs[4])
        self._pub_joint_cmds["l_elbow"].publish(angs[5])
    
    def publish_leg_cmds(self,angs):
        # the function publishes the commanded angles to the corresponding publisher topics:
        self._pub_joint_cmds["r_hip_frontal"].publish(angs[0])
        self._pub_joint_cmds["l_hip_frontal"].publish(angs[1])
        self._pub_joint_cmds["r_hip_sagittal"].publish(angs[2])
        self._pub_joint_cmds["l_hip_sagittal"].publish(angs[3])
        self._pub_joint_cmds["r_knee"].publish(angs[4])
        self._pub_joint_cmds["l_knee"].publish(angs[5]) 
        self._pub_joint_cmds["r_ankle_sagittal"].publish(angs[6])
        self._pub_joint_cmds["l_ankle_sagittal"].publish(angs[7]) 
        self._pub_joint_cmds["r_ankle_frontal"].publish(angs[8])
        self._pub_joint_cmds["l_ankle_frontal"].publish(angs[9])
    
    def publish_sag_cmds(self,angs):
        # the function publishes the commanded angles to the corresponding publisher topics:
        self._pub_joint_cmds["r_shoulder_sagittal"].publish(angs[0])
        self._pub_joint_cmds["l_shoulder_sagittal"].publish(angs[1])
        self._pub_joint_cmds["r_hip_sagittal"].publish(angs[2])
        self._pub_joint_cmds["l_hip_sagittal"].publish(angs[3])
        self._pub_joint_cmds["r_knee"].publish(angs[4])
        self._pub_joint_cmds["l_knee"].publish(angs[5]) 
        self._pub_joint_cmds["r_ankle_sagittal"].publish(angs[6])
        self._pub_joint_cmds["l_ankle_sagittal"].publish(angs[7])        

    def publish_leg_angs(self,angs):
        # the function publishes the actual joint angles to the corresponding publisher topics:
        self._pub_joints["r_hip_frontal"].publish(angs[0])
        self._pub_joints["l_hip_frontal"].publish(angs[1])
        self._pub_joints["r_hip_sagittal"].publish(angs[2])
        self._pub_joints["l_hip_sagittal"].publish(angs[3])
        self._pub_joints["r_knee"].publish(angs[4])
        self._pub_joints["l_knee"].publish(angs[5]) 
        self._pub_joints["r_ankle_sagittal"].publish(angs[6])
        self._pub_joints["l_ankle_sagittal"].publish(angs[7]) 
        self._pub_joints["r_ankle_frontal"].publish(angs[8])
        self._pub_joints["l_ankle_frontal"].publish(angs[9]) 

    def publish_sag_angs(self,angs):
        # the function publishes the actual joint angles to the corresponding publisher topics:
        self._pub_joints["r_shoulder_sagittal"].publish(angs[0])
        self._pub_joints["l_shoulder_sagittal"].publish(angs[1])
        self._pub_joints["r_hip_sagittal"].publish(angs[2])
        self._pub_joints["l_hip_sagittal"].publish(angs[3])
        self._pub_joints["r_knee"].publish(angs[4])
        self._pub_joints["l_knee"].publish(angs[5]) 
        self._pub_joints["r_ankle_sagittal"].publish(angs[6])
        self._pub_joints["l_ankle_sagittal"].publish(angs[7]) 

    def publish_leg_ang_vels(self,vels):
        # the function publishes the joint angular velocity estimates to the corresponding publisher topics:
        self._pub_joint_vels["r_hip_frontal"].publish(vels[0])
        self._pub_joint_vels["l_hip_frontal"].publish(vels[1])
        self._pub_joint_vels["r_hip_sagittal"].publish(vels[2])
        self._pub_joint_vels["l_hip_sagittal"].publish(vels[3])
        self._pub_joint_vels["r_knee"].publish(vels[4])
        self._pub_joint_vels["l_knee"].publish(vels[5]) 
        self._pub_joint_vels["r_ankle_sagittal"].publish(vels[6])
        self._pub_joint_vels["l_ankle_sagittal"].publish(vels[7]) 
        self._pub_joint_vels["r_ankle_frontal"].publish(vels[8])
        self._pub_joint_vels["l_ankle_frontal"].publish(vels[9]) 

    def publish_sag_ang_vels(self,vels):
        # the function publishes the joint angular velocity estimates to the corresponding publisher topics:
        self._pub_joint_vels["r_shoulder_sagittal"].publish(vels[0])
        self._pub_joint_vels["l_shoulder_sagittal"].publish(vels[1])
        self._pub_joint_vels["r_hip_sagittal"].publish(vels[2])
        self._pub_joint_vels["l_hip_sagittal"].publish(vels[3])
        self._pub_joint_vels["r_knee"].publish(vels[4])
        self._pub_joint_vels["l_knee"].publish(vels[5]) 
        self._pub_joint_vels["r_ankle_sagittal"].publish(vels[6])
        self._pub_joint_vels["l_ankle_sagittal"].publish(vels[7]) 
        
    def convert_angles_to_commands(self,ids,angles):
        # this function converts an array of angle values (in radians) to the corresponding 10-bit motor values,
        # assuming that the 'ids' array contains matching ID numbers for the angle values of 'angles' array
        
        b=[60,240,60,240,150,150,150,150,150,150,150,150,150,150,150,150,150,150] # motor offsets
        c=[1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1] # motor polarities
        
        cmds=np.zeros(len(ids)) # initialize cmds array
        for j in range(len(ids)): # get 10-bit motor values:
            num=ids[j] # get motor ID number from array
            ang=angles[j] # get desired joint angle value in radians
            cmds[j]=(1023/300)*(ang*((180/math.pi)*c[num-1])+b[num-1]) # convert to degrees, apply polarity, convert to 10-bit, add offset
        return cmds

    def convert_motor_positions_to_angles(self,ids,cmds):
        # this function converts an array of 10-bit motor values to the corresponding angle values (in radians),
        # assuming that the 'ids' array contains matching ID numbers for the angle values of 'cmds' array
        
        b=[60,240,60,240,150,150,150,150,150,150,150,150,150,150,150,150,150,150] # motor offsets
        c=[1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1] # motor polarities
        
        angs=np.zeros(len(ids)) # initialize cmds array
        for j in range(len(ids)): # get 10-bit motor values:
            num=ids[j] # get motor ID number from array
            cmd=cmds[j] # get motor position as 10-bit value
            angs[j]=(((300/1023)*cmd)-b[num-1])*c[num-1]*(math.pi/180) # convert from 10-bit, subtract offset, apply polarity, go to radians
        return angs    
