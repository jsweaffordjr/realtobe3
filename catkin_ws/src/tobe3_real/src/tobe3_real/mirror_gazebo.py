#!/usr/bin/python3

from threading import Thread
import rospy
import math
import numpy as np
import random
import torch
import torch.nn as nn
from numpy import array
from numpy.linalg import norm
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from geometry_msgs.msg import Quaternion
from tobe3_real.tobe import Tobe
from torch.distributions.beta import Beta

"""
    This code allows for the real TOBE to (attempt to) mirror what happens with the Gazebo TOBE.
    The commands that are published and run in Gazebo are mimicked on the actual robot. 
"""

def HOSM_diff(dt,signal,est_signal,est_deriv,est_deriv2,est_deriv3):
# 1-step homogeneous discrete sliding-mode-based differentiator, assuming 2 derivatives (z0dot and z1dot) are needed:
# signal, est_deriv, est_deriv2, and est_deriv3 can be arrays/lists/vectors, but need to be same length
# inputs: signal-- value/list being differentiated, dt--estimated timestep between incoming values
    L=5 # Lipschitz constant

    qz = np.array(signal)
    z0 = np.array(est_signal)
    z1 = np.array(est_deriv)
    z2 = np.array(est_deriv2)
    z3 = np.array(est_deriv3)
    z0dot=z1-3*(L**(1/4))*(abs(z0-qz)**(3/4))*np.sign(z0-qz)
    z1dot=z2-4.16*(L**(1/2))*(abs(z0-qz)**(1/2))*np.sign(z0-qz)
    z2dot=z3-3.06*(L**(3/4))*(abs(z0-qz)**(1/4))*np.sign(z0-qz)
    z3dot=-1.1*L*np.sign(z0-qz)
    z0_next=z0+dt*z0dot+0.5*dt*dt*z2+(1/6)*(dt*dt*dt*z3)
    z1_next=z1+dt*z1dot+0.5*dt*dt*z3
    z2_next=z2+dt*z2dot
    z3_next=z3+dt*z3dot
    
    return z0dot, z1dot, z0_next, z1_next, z2_next, z3_next
    
class Stand:
    """
    Class for making Tobe stand
    """

    def __init__(self,tobe):
        self.tobe=tobe
    
        # set up 'calib' subscriber for pause in Start function until calibration is finished
        self.calib=False
        rospy.sleep(3)
        self.sub_cal=rospy.Subscriber('/calibration',Vector3,self._calibrate, queue_size=1)
    
        # initialize standing thread:
        self._th_stand = None

        # variables, parameters:
        self.dt=np.round(1.0/6.0, decimals=3)
        self.active = False
        self.fore_lean=[0,0,0,0,0] # last 5 'fore_lean' angle values
        self.side_lean=[0,0,0,0,0] # last 5 'side_lean' angle values
        self.f_deriv=0
        self.f_ddot=0
        self.f_int=0
        self.s_deriv=0
        self.s_ddot=0
        self.s_int=0
        self.fore_data=Vector3()
        self.side_data=Vector3()
        
        # differentiation arrays for torso angles:
        self.z0_next=[0,0]
        self.z1_next=[0,0]
        self.z2_next=[0,0]
        self.z3_next=[0,0]
        
        # differentiation arrays for joint angles:
        self.q0_next=[0,0,0,0,0,0,0,0,0,0]
        self.q1_next=[0,0,0,0,0,0,0,0,0,0]
        self.q2_next=[0,0,0,0,0,0,0,0,0,0]
        self.q3_next=[0,0,0,0,0,0,0,0,0,0]
        
        # reset arrays for joint angles (used to reset differentiators when joint angle values don't change within 5 timesteps):
        self.q0_last1=[0,0,0,0,0,0,0,0,0,0]
        self.q0_last2=[0,0,0,0,0,0,0,0,0,0]
        self.q0_last3=[0,0,0,0,0,0,0,0,0,0]
        self.q0_last4=[0,0,0,0,0,0,0,0,0,0]

        # thresholds:
        self.lean_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.az_min = 0.1 # push threshold: values less than this are ignored
        
        # subscribers and publishers:
        self._sub_quat = rospy.Subscriber("/lean", Vector3, self._update_orientation, queue_size=5) # subscribe to lean topic
        self.leandata1 = rospy.Publisher('leandata1', Vector3, queue_size=1)
        self.leandata2 = rospy.Publisher('leandata2', Vector3, queue_size=1)
        
        self.TOBE_cmds=[0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18]
        
        # joint command subscribers:
        self.j01 = rospy.Subscriber("/tobe/r_hip_lateral_joint_position_controller/command", Float64, self._updatej01, queue_size=5)
        self.j02 = rospy.Subscriber("/tobe/l_hip_lateral_joint_position_controller/command", Float64, self._updatej02, queue_size=5)
        self.j03 = rospy.Subscriber("/tobe/r_hip_swing_joint_position_controller/command", Float64, self._updatej03, queue_size=5)
        self.j04 = rospy.Subscriber("/tobe/l_hip_swing_joint_position_controller/command", Float64, self._updatej04, queue_size=5)
        self.j05 = rospy.Subscriber("/tobe/r_knee_joint_position_controller/command", Float64, self._updatej05, queue_size=5)
        self.j06 = rospy.Subscriber("/tobe/l_knee_joint_position_controller/command", Float64, self._updatej06, queue_size=5)
        self.j07 = rospy.Subscriber("/tobe/r_ankle_swing_joint_position_controller/command", Float64, self._updatej07, queue_size=5)
        self.j08 = rospy.Subscriber("/tobe/l_ankle_swing_joint_position_controller/command", Float64, self._updatej08, queue_size=5)
        self.j09 = rospy.Subscriber("/tobe/r_ankle_lateral_joint_position_controller/command", Float64, self._updatej09, queue_size=5)
        self.j10 = rospy.Subscriber("/tobe/l_ankle_lateral_joint_position_controller/command", Float64, self._updatej10, queue_size=5)
        
        # switch to 'active' status, move to initial 'home' position, if not already there:
        self.start()       
                    
    def _update_orientation(self, msg):
        """
        Catches lean angle data and updates robot orientation
        """
        fore = msg.z # get z-(i.e., forward/backward direction)component of initially vertical x-axis
        side = msg.y # get y-(i.e., left/right direction)component of initially vertical x-axis
        if abs(fore) < self.lean_min: # apply threshold
            fore = 0
            if sum(self.fore_lean[1:]) == 0:
                self.f_int = 0 # reset integrator if past five values are zero
                self.z0_next[0]=0.0 # reset differentiators also...
                self.z1_next[0]=0.0
                self.z2_next[0]=0.0
        if abs(side) < self.lean_min: # apply threshold
            side = 0
            if sum(self.side_lean[1:]) == 0:
                self.s_int = 0 # reset integrator if past five values are zero
                self.z0_next[1]=0.0 # reset differentiators also...
                self.z1_next[1]=0.0
                self.z2_next[1]=0.0
                
        #qz=fore # use lean factor instead of lean angle
        #qy=side # use lean factor instead of lean angle
        qz = math.asin(-fore) # convert lean factor to lean angle (inverse sine of z-component of IMU x-axis [which is a unit vector])   
        qy = math.asin(-side) # convert lean factor to lean angle (inverse sine of y-component of IMU x-axis [which is a unit vector]) 
        
        self.fore_lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.side_lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.fore_lean.append(qz) # append newest value to end of the array
        self.side_lean.append(qy) # append newest value to end of the array
   
        # derivative and integral estimates:
        dt=self.dt # approximate dt between data points
        
        # update integrals:
        area1=0.5*dt*(self.fore_lean[3]+self.fore_lean[4]) # trapezoidal integration between two values
        prev1=self.f_int # get previous value of integral
        self.f_int=prev1+area1 # updated integral value
        
        area2=0.5*dt*(self.side_lean[3]+self.side_lean[4]) # trapezoidal integration between two values
        prev2=self.s_int # get previous value of integral
        self.s_int=prev2+area2 # updated integral value

        # HDD output:
        q = [qz,qy]
        z0 = self.z0_next
        z1 = self.z1_next
        z2 = self.z2_next
        z3 = self.z3_next
        [z0dot, z1dot,self.z0_next,self.z1_next,self.z2_next,self.z3_next] = HOSM_diff(dt, q, z0, z1, z2, z3)
        self.f_deriv = (qz - z0[0])/self.dt
        self.s_deriv = (qy - z0[1])/self.dt
        self.f_ddot = z1dot[0]
        self.s_ddot = z1dot[1]
        self.z0_next = q
        
        # assign lean angle and derivative(s) and/or integral:
        self.fore_data.x=qz
        self.fore_data.y=self.f_deriv
        self.fore_data.z=self.f_ddot
        self.leandata1.publish(self.fore_data)
        self.side_data.x=qy
        self.side_data.y=self.s_deriv
        self.side_data.z=self.s_ddot
        self.leandata2.publish(self.side_data)

    
    # joint update callbacks:     
    def _updatej01(self, msg):
        self.TOBE_cmds[0] = msg.data 
    def _updatej02(self, msg):
        self.TOBE_cmds[1] = msg.data
    def _updatej03(self, msg):
        self.TOBE_cmds[2] = msg.data        
    def _updatej04(self, msg):
        self.TOBE_cmds[3] = msg.data 
    def _updatej05(self, msg):
        self.TOBE_cmds[4] = msg.data
    def _updatej06(self, msg):
        self.TOBE_cmds[5] = msg.data        
    def _updatej07(self, msg):
        self.TOBE_cmds[6] = msg.data 
    def _updatej08(self, msg):
        self.TOBE_cmds[7] = msg.data
    def _updatej09(self, msg):
        self.TOBE_cmds[8] = msg.data        
    def _updatej10(self, msg):
        self.TOBE_cmds[9] = msg.data 


    def _calibrate(self,msg):
        x=msg.x
        y=msg.y
        z=msg.z
        if x > 2 and y > 2 and z > 2:
            self.calib=True

    def start(self):
        if not self.active:
            self.active = True
            self.tobe.turn_on_motors()
            self.init_stand()
            
            # pause until 10 seconds after IMU is calibrated:
            while self.calib == False:
                rospy.sleep(0.1)
            rospy.sleep(10) 
            
            # start standing control loop
            self._th_stand = Thread(target=self._do_stand)
            self._th_stand.start()
            self.standing = True
                                           
    def init_stand(self):
        """
        If not already there yet, go to initial standing position
        """
        rospy.loginfo("Going to initial stance")
        cmd=[281,741,272,750,409,613,511,511,543,480,609,414,172,851,295,728,547,476]
        self.tobe.command_all_motors(cmd)

        # go to initial leg configuration
        #leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        #init_leg_config = [0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18]
        #cmd2 = self.tobe.convert_angles_to_commands(leg_angle_ids,init_leg_config)
        #self.tobe.command_leg_motors(cmd2)
        #rospy.loginfo("Initial leg configuration: ")
        #rospy.loginfo(cmd2)
        #rospy.loginfo("Now at initial stance position.")
   
    def _do_stand(self):
        """
        Main standing control loop
        """
        dt=self.dt # approximate dt between data points
        samplerate = 6
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread")

        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
               
        elapsed_time = 0

        #refs_tobe = [0.17,-0.17,-0.5,0.5,1.75,-1.75,1.12,-1.12,0.17,-0.17] # in Gazebo
        #refs_tobe = self.tobe.convert_motor_positions_to_angles(leg_angle_ids,self.tobe.read_leg_motor_positions())
        #lower_bounds = [-0.3,-0.5,-0.5,-0.5,-0.3,-0.3,-0.5,-0.5,-0.5,-0.3] # in MuJoCo
        #upper_bounds = [0.3,0.5,0.5,0.5,0.3,0.3,0.5,0.5,0.5,0.3] # in MuJoCo

        # order of lower, upper bounds is based on Mujoco sim: 
        # [Rhip_lat, Rhip_sw, Rknee, Rankle_sw, Rankle_lat, Lhip_lat, Lhip_sw, Lknee, Lankle_sw, Lankle_lat]

        while not rospy.is_shutdown(): 
            # read joint motor positions:
            leg_angs = self.tobe.convert_motor_positions_to_angles(leg_angle_ids,self.tobe.read_leg_motor_positions())
            self.tobe.publish_leg_angs(leg_angs) # publish leg joint angles
         
            torso_ang = [self.fore_data.x, self.side_data.x] # get torso angle vector
            torso_ang_vel = [self.fore_data.y, self.side_data.y] # get torso angular velocity vector
            
            # execute commands:
            leg_ang_cmds = self.tobe.convert_angles_to_commands(leg_angle_ids, self.TOBE_cmds)
            self.tobe.command_leg_motors(leg_ang_cmds)
            
            # on-screen feedback:
            #rospy.loginfo(self.TOBE_cmds) 
            #rospy.loginfo(leg_angs)
            
            # send commands to motors and record command angle (in radians)
            #self.tobe.command_leg_motors(response) # send 10-bit joint commands to motors
            self.tobe.publish_leg_cmds(self.TOBE_cmds) # send joint angle commands to publisher topics  
            r.sleep()
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None









