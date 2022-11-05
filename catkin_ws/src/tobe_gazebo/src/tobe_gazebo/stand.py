#!/usr/bin/python3

from threading import Thread
import rospy
import math
import numpy as np
import random
from numpy import array
from numpy.linalg import norm
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from geometry_msgs.msg import Quaternion
from tobe_gazebo.tobe import Tobe

class StandFunc:
    """
    Stand Function
    Provides parameters for standing
    """

    def __init__(self):
        # array of joint names, in order from kinematic chain 
        j=["l_ankle_frontal","l_ankle_sagittal","l_knee","l_hip_sagittal","l_hip_frontal",
                "l_hip_swivel","r_hip_swivel","r_hip_frontal","r_hip_sagittal","r_knee","r_ankle_sagittal",
                "r_ankle_frontal","l_shoulder_sagittal","l_shoulder_frontal","l_elbow","r_shoulder_sagittal","r_shoulder_frontal","r_elbow"]
        
        # array of initial joint angle commands
        f = [-0.2, -1.1932, -1.7264, 0.4132, -0.15, 0, 0, 0.15, -0.4132, 1.7264,
             1.1932, 0.2, -0.3927, -0.3491, -0.5236, 0.3927, -0.3491, 0.5236]
             
        # convert joint angle, name order to ROBOTIS right/left (odd-/even-numbered) from head to toe:     
        self.init_angles = [f[15],f[12],f[16],f[13],f[17],f[14],f[6],f[5],f[7],f[4],f[8],f[3],f[9],f[2],f[10],f[1],f[11],f[0]]
        self.joints = [j[15],j[12],j[16],j[13],j[17],j[14],j[6],j[5],j[7],j[4],j[8],j[3],j[9],j[2],j[10],j[1],j[11],j[0]]        
        
        # thresholds:
        self.qx_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.ax_min = 0.1 # push threshold: values less than this are ignored
    
    def recover(self, push, push_threshold,gain1,gain2):
        max_push = 5 # value of max expected torso acceleration, in m/s^2
        move = [0,0]
        max_response = [gain1,gain2] # max responses of shoulders, hips, respectively
        
        if push >= max_push: # if push exceeds maximum expected, 
            move = max_response # set response to maximum
        elif push > push_threshold: # if push exceeds threshold, set mid response:
            move1 = 0.7*max_response[0]
            move2 = 0.7*max_response[1]
            move = [move1,move2]
        
        return move
        
    def get_angles(self, z_lean, l_deriv, l_ddot, sag_ang_diffs, falling):
        # recall initial joint angles  
        f = [0.3927,-0.3927,-0.3491,-0.3491,0.5236,-0.5236,0,0,0.2,-0.2,-0.45,0.45,1.7264,-1.7264,1.1932,-1.1932,0.21,-0.21]
        cmds=np.zeros(8) # initialize output array
        
        if not falling:
            # use ankle or hip/shoulder ctrl
            if sag_ang_diffs[0] != 0 or sag_ang_diffs[1] != 0:
                diff1 = sag_ang_diffs[0]
                diff2 = sag_ang_diffs[1]
                diff4 = 0
            else:
                # ankle controller gains
                Kpa = 0.005 # proportional gain for ankles    
                Kda = 0 # derivative gain for ankles
            
                # compute ankle adjustments:
                h = 0.29 # height of IMU from ground
                g = 9.81 # acceleration due to gravity
                omega_inv = math.sqrt(h/g)
                diff4 = Kpa*(z_lean[4]+omega_inv*l_deriv) + Kda*(l_deriv+omega_inv*l_ddot)  # PD control of ankles (capture point) 
                diff1 = 0
                diff2 = 0
        
            # left ankle angle should increase (f15 + diff), right ankle angle should decrease (f14 - diff) when forward lean occurs
            # left shoulder angle should increase (f1 + diff3), right shoulder angle should decrease (f0 - diff3) when forward acc. occurs
            # left hip angle should increase (f11 + diff2), right hip angle should decrease (f10 - diff2) when forward acceleration occurs
            f1 = self.init_angles[14] #- diff4 # right sagittal ankle
            f2 = self.init_angles[15] #+ diff4 # left sagittal ankle
            self.init_angles[14] = f1
            self.init_angles[15] = f2

            cmds[0] = f[0] #- diff1 # right sagittal shoulder
            cmds[1] = f[1] #+ diff1 # left sagittal shoulder
            cmds[2] = f[10] #- diff2 # right sagittal hip
            cmds[3] = f[11] #+ diff2 # left sagittal hip
            cmds[4] = f[12]  # right knee
            cmds[5] = f[13]  # left knee
            cmds[6] = f1 # right sagittal ankle
            cmds[7] = f2 # left sagittal ankle
        
        else:
            self.init_angles = f
            diff1 = sag_ang_diffs[0]
            diff2 = sag_ang_diffs[1]
            diff3 = sag_ang_diffs[2]
            diff4 = sag_ang_diffs[3]
            
            cmds[0] = f[0] - diff1 # right sagittal shoulder
            cmds[1] = f[1] + diff1 # left sagittal shoulder
            cmds[2] = f[10] - diff2 # right sagittal hip
            cmds[3] = f[11] + diff2 # left sagittal hip
            cmds[4] = f[12] + diff3 # right knee
            cmds[5] = f[13] - diff3 # left knee
            cmds[6] = f[14] + diff4 # right sagittal ankle
            cmds[7] = f[15] - diff4 # left sagittal ankle

        return cmds
               

class Stand:
    """
    Class for making Tobe stand
    """

    def __init__(self,tobe):
        self.func = StandFunc()
        self.tobe=tobe
    
        # initialization parameters:
        self.active = False
        self.standing = False
        self.ready_pos = self.func.init_angles
        self._th_stand = None
        self.response = [281,741,610,412,297,726]
        self.prev_response = self.response

        # other variables, parameters:
        self.push=0 # smoothed z-acceleration value
        self.lean=[0,0,0,0,0] # last 5 values from 'lean' publisher
        self.l_deriv=0
        self.l_ddot=0
        self.l_int=0
        self.ldata=Vector3()
        self.x_next=[0,0,0,0]
        self.var1=0.8
        self.var2=0.2
        self.qx_min = self.func.qx_min # lean threshold: values less than this are rounded down to zero
        self.ax_min = self.func.ax_min # push threshold: values less than this are rounded down to zero
        
        # push-recovery-related parameters:
        self.home_time=0 # time until hips, shoulders return to home position
        self.responding=False # denotes when robot is responding quickly to disturbance or fall
        self.going_home=False # denotes when robot is returning to home position after disturbance response or fall
        self.reflex=[0,0,0,0] # difference bet. reflex and home joint positions when responding to disturbance
        self.push_threshold=2.5 # threshold above which push recovery responses occur
        self.lean_threshold=0.3 # threshold above which imminent fall is detected
        
        # subscribers and publishers:
        self._sub_quat = rospy.Subscriber("/lean", Vector3, self._update_orientation, queue_size=5) # subscribe to lean topic
        self._sub_acc = rospy.Subscriber("/push", Vector3, self._update_acceleration, queue_size=5) # subscribe to push topic
        self.leandata = rospy.Publisher('leandata', Vector3, queue_size=1)
        self.pushdata = rospy.Publisher('pushdata', Float64, queue_size=1)
        self.Ktuning = rospy.Publisher('var1', Float64, queue_size=1)
        self.Ktuning2 = rospy.Publisher('var2', Float64, queue_size=1)
        self.app_force = rospy.Publisher('rand_force', Float64, queue_size=1)
        self._sub_gain = rospy.Subscriber("/var1", Float64, self._update_gain1, queue_size=5)
        self._sub_gain2 = rospy.Subscriber("/var2", Float64, self._update_gain2, queue_size=5)
        
        # switch to 'active' status, move to initial 'home' position, if not already there:
        self.start()

    def _update_gain1(self, msg):
        # updates 'var1' during experiment
        self.var1 = msg.data
        
    def _update_gain2(self, msg):
        # updates 'var2' during experiment
        self.var2 = msg.data
           
    def _update_acceleration(self, msg):
        """
        Catches acceleration data and applies exponentially weighted moving average to smooth out data output
        """
        w = 1 # weight of current data point's contribution to moving average output
        ax = msg.x # get acceleration in x-direction      
        if abs(ax) < self.ax_min: # apply threshold
            ax = 0    
        acc=w*ax+(1-w)*self.push # update exponentially weighted moving average
        self.pushdata.publish(acc) # publish current (smoothed) acceleration value
        self.push=acc # save current value in 'push'
        
        # respond to push that causes acceleration greater than self.push_threshold:
        if ((acc > self.push_threshold) and not self.going_home and self.standing):
            self.responding = True
                    
    def _update_orientation(self, msg):
        """
        Catches lean angle data and updates robot orientation
        """
        q = msg.x # get x-(i.e., forward/backward direction)component of initially vertical z-axis
        if abs(q) < self.qx_min: # apply threshold
            q = 0
            if self.lean[1]+self.lean[2]+self.lean[3]+self.lean[4] == 0:
                self.l_int = 0 # reset integrator if past five values are zero
        
        #qx=q # use lean factor instead of lean angle
        qx = math.asin(q) # convert lean factor to lean angle (inverse sine of x-component of IMU z-axis [which is a unit vector])   
        self.lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.lean.append(qx) # append newest value to end of the array
   
        # derivative and integral estimates:
        dt=0.02 # approximate dt between data points
        
        area=0.5*dt*(self.lean[3]+self.lean[4]) # trapezoidal integration between two values
        prev=self.l_int # get previous value of integral
        self.l_int=prev+area # updated integral value
        
        # homogeneous discrete sliding-mode-based differentiator:
        L=5 # Lipschitz constant
        
        # option 1: if only 1 derivative (x0dot) is needed, use this:
        #x0=self.x_next[0]
        #x1=self.x_next[1]
        #x2=self.x_next[2]
        #x0dot=x1-2.12*(L**(1/3))*(abs(x0-qx)**(2/3))*np.sign(x0-qx)
        #x1dot=x2-2*(L**(2/3))*(abs(x0-qx)**(1/3))*np.sign(x0-qx)
        #x2dot=-1.1*L*np.sign(x0-qx)
        #self.x_next[0]=x0+dt*x0dot+0.5*dt*dt*x2
        #self.x_next[1]=x1+dt*x1dot
        #self.x_next[2]=x2+dt*x2dot
        
        # option 2: if 2 derivatives (x0dot and x1dot) are needed, use this:
        x0=self.x_next[0]
        x1=self.x_next[1]
        x2=self.x_next[2]
        x3=self.x_next[3]
        x0dot=x1-3*(L**(1/4))*(abs(x0-qx)**(3/4))*np.sign(x0-qx)
        x1dot=x2-4.16*(L**(1/2))*(abs(x0-qx)**(1/2))*np.sign(x0-qx)
        x2dot=x3-3.06*(L**(3/4))*(abs(x0-qx)**(1/4))*np.sign(x0-qx)
        x3dot=-1.1*L*np.sign(x0-qx)
        self.x_next[0]=x0+dt*x0dot+0.5*dt*dt*x2+(1/6)*(dt*dt*dt*x3)
        self.x_next[1]=x1+dt*x1dot+0.5*dt*dt*x3
        self.x_next[2]=x2+dt*x2dot
        self.x_next[3]=x3+dt*x3dot

        # HDD output:
        self.l_deriv = x0dot 
        self.l_ddot = x1dot # only use this if using option 2
        
        # assign lean angle and derivative(s) and/or integral:
        self.ldata.x=qx
        self.ldata.y=self.l_deriv
        self.ldata.z=self.l_ddot
        self.leandata.publish(self.ldata)


    def start(self):
        if not self.active:
            self.active = True
            self.init_stand()
            #
            
            # start standing control loop
            self._th_stand = Thread(target=self._do_stand)
            self._th_stand.start()
            self.standing = True
                                           
    def init_stand(self):
        """
        If not already there yet, go to initial standing position
        """
        
        rospy.loginfo("Now at initial stance position.")
        rospy.sleep(10)
        rospy.loginfo("Control loop starts now.")
        #<!---J r_shoulder_swing_joint 0.3927 -J l_shoulder_swing_joint -0.3927 -J r_shoulder_lateral_joint -0.3491 -J l_shoulder_lateral_joint -0.3491 -J r_elbow_joint 0.5236 -J l_elbow_joint -0.5236 -J r_hip_lateral_joint 0.15 -J l_hip_lateral_joint -0.15 -J r_hip_swing_joint -0.45 -J l_hip_swing_joint 0.45 -J r_knee_joint 1.7264 -J l_knee_joint -1.7264 -J r_ankle_swing_joint 1.1932 -J l_ankle_swing_joint -1.1932 -J r_ankle_lateral_joint 0.2 -J l_ankle_lateral_joint -0.2--> 

    def _do_stand(self):
        """
        Main standing control loop
        """
        samplerate = 60
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread")
        func = self.func
        sag_angle_ids = [1,2,11,12,13,14,15,16]
               
        while not rospy.is_shutdown(): 
            # read joint motor positions:
            sag_angs = self.tobe.read_sag_angles()
            reflex_return = [0,0,0,0]
            
            if self.standing: # standing control loop  
                reflex_time = 0.3 # reflex response time
                return_time = 0.2 # period of time between reflex and return to home position
                pause_at_home_time = 1 # pause time at home position before being ready to respond again

                # time-based control logic:
                if self.going_home:
                    self.home_time = self.home_time - (1/samplerate) # decrement 'home_time'
                    if self.home_time <= 0: # end of reflex period
                        self.going_home = False
                        rospy.loginfo("Finished responding to push.")
                    else:
                        if self.home_time > (pause_at_home_time + return_time): # during reflex response
                            reflex_return = self.reflex      
                        else:
                            if self.home_time > pause_at_home_time: # returning to home position
                                reflex_return1 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[0]
                                reflex_return2 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[1]
                                reflex_return3 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[2]
                                reflex_return4 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[3]
                                reflex_return = [reflex_return1, reflex_return2, reflex_return3, reflex_return4]                             
                
                # set reflex and total time until recovery:
                if self.responding:
                    reflex1 = func.recover(self.push,self.push_threshold,self.var1,self.var2)  
                    self.reflex = [reflex1[0], reflex1[1], 0, 0]
                    self.responding = False
                    self.going_home = True
                    self.home_time = reflex_time + return_time + pause_at_home_time
                    reflex_return = self.reflex 
                    rospy.loginfo("Responding to a push...")
                                          
            # compute appropriate joint commands and execute commands:        
            self.response=func.get_angles(self.lean,self.l_deriv,self.l_ddot,reflex_return,not self.standing)
            self.tobe.command_sag_motors(self.response) # send joint commands to motors
            r.sleep()
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

