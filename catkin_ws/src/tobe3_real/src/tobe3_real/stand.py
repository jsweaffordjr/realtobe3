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
from tobe3_real.tobe import Tobe

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
        f = [-0.2, -1.1, -1.7264, 0.5064, -0.15, 0, 0, 0.15, -0.5064, 1.7264,
             1.1, 0.2, -0.3927, -0.3491, -0.5236, 0.3927, -0.3491, 0.5236]
        
        # convert joint angle, name order to ROBOTIS right/left (odd-/even-numbered) from head to toe:     
        self.init_angles = [f[15],f[12],f[16],f[13],f[17],f[14],f[6],f[5],f[7],f[4],f[8],f[3],f[9],f[2],f[10],f[1],f[11],f[0]]
        self.joints = [j[15],j[12],j[16],j[13],j[17],j[14],j[6],j[5],j[7],j[4],j[8],j[3],j[9],j[2],j[10],j[1],j[11],j[0]]     
        
    def get_angles(self, z_lean, l_deriv, l_ddot, sag_ang_diffs, falling):
        # recall initial joint angles  
        f = [0.3927,-0.3927,-0.3491,-0.3491,0.5236,-0.5236,0,0,0.15,-0.15,-0.5064,0.5064,1.7264,-1.7264,1.1,-1.1,0.2,-0.2]
        cmds=np.zeros(8) # initialize output array
        
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
    
        # set up 'calib' subscriber for pause in Start function until calibration is finished
        self.calib=False
        rospy.sleep(3)
        self.sub_cal=rospy.Subscriber('/calibration',Vector3,self._calibrate, queue_size=1)
    
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
        self.z_next=[0,0,0,0]
        self.var1=0.8
        self.var2=0.2
        self.qz_min = self.func.qz_min # lean threshold: values less than this are rounded down to zero
        self.az_min = self.func.az_min # push threshold: values less than this are rounded down to zero
        
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
        az = msg.z # get acceleration in z-direction      
        if abs(az) < self.az_min: # apply threshold
            az = 0    
        acc=w*az+(1-w)*self.push # update exponentially weighted moving average
        self.pushdata.publish(acc) # publish current (smoothed) acceleration value
        self.push=acc # save current value in 'push'
        
        # respond to push that causes acceleration greater than self.push_threshold:
        if ((acc > self.push_threshold) and not self.going_home and self.standing):
            self.responding = True
                    
    def _update_orientation(self, msg):
        """
        Catches lean angle data and updates robot orientation
        """
        q = msg.z # get z-(i.e., forward/backward direction)component of initially vertical x-axis
        if abs(q) < self.qz_min: # apply threshold
            q = 0
            if self.lean[1]+self.lean[2]+self.lean[3]+self.lean[4] == 0:
                self.l_int = 0 # reset integrator if past five values are zero
        
        #qz=q # use lean factor instead of lean angle
        qz = math.asin(q) # convert lean factor to lean angle (inverse sine of z-component of IMU x-axis [which is a unit vector])   
        self.lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.lean.append(qz) # append newest value to end of the array
   
        # derivative and integral estimates:
        dt=0.02 # approximate dt between data points
        
        area=0.5*dt*(self.lean[3]+self.lean[4]) # trapezoidal integration between two values
        prev=self.l_int # get previous value of integral
        self.l_int=prev+area # updated integral value
        
        # homogeneous discrete sliding-mode-based differentiator:
        L=5 # Lipschitz constant
        
        # option 1: if only 1 derivative (z0dot) is needed, use this:
        #z0=self.z_next[0]
        #z1=self.z_next[1]
        #z2=self.z_next[2]
        #z0dot=z1-2.12*(L**(1/3))*(abs(z0-qz)**(2/3))*np.sign(z0-qz)
        #z1dot=z2-2*(L**(2/3))*(abs(z0-qz)**(1/3))*np.sign(z0-qz)
        #z2dot=-1.1*L*np.sign(z0-qz)
        #self.z_next[0]=z0+dt*z0dot+0.5*dt*dt*z2
        #self.z_next[1]=z1+dt*z1dot
        #self.z_next[2]=z2+dt*z2dot
        
        # option 2: if 2 derivatives (z0dot and z1dot) are needed, use this:
        z0=self.z_next[0]
        z1=self.z_next[1]
        z2=self.z_next[2]
        z3=self.z_next[3]
        z0dot=z1-3*(L**(1/4))*(abs(z0-qz)**(3/4))*np.sign(z0-qz)
        z1dot=z2-4.16*(L**(1/2))*(abs(z0-qz)**(1/2))*np.sign(z0-qz)
        z2dot=z3-3.06*(L**(3/4))*(abs(z0-qz)**(1/4))*np.sign(z0-qz)
        z3dot=-1.1*L*np.sign(z0-qz)
        self.z_next[0]=z0+dt*z0dot+0.5*dt*dt*z2+(1/6)*(dt*dt*dt*z3)
        self.z_next[1]=z1+dt*z1dot+0.5*dt*dt*z3
        self.z_next[2]=z2+dt*z2dot
        self.z_next[3]=z3+dt*z3dot

        # HDD output:
        self.l_deriv = z0dot 
        self.l_ddot = z1dot # only use this if using option 2
        
        # assign lean angle and derivative(s) and/or integral:
        self.ldata.x=qz
        self.ldata.y=self.l_deriv
        self.ldata.z=self.l_ddot
        self.leandata.publish(self.ldata)
        
        # assume imminent fall when lean angle exceeds self.lean_threshold:
        if ((qz > self.lean_threshold) and self.standing):
            self.standing = False
            self.responding = True
            self.going_home = False


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
        cmd=[281,741,272,750,409,613,511,511,540,482,610,412,174,848,297,726,550,472]
        self.tobe.command_all_motors(cmd)
        rospy.loginfo("Now at initial stance position.")

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
            sag_angs = self.tobe.convert_motor_positions_to_angles(sag_angle_ids,self.tobe.read_sag_motor_positions())
            self.tobe.publish_sag_angs(sag_angs)
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

            else: # falling behavior
                squat_time = 8 # robot moves to squat position while down within this time
                return_time = 0.8 # robot returns to home position during this time
                pause_at_home_time = 1.2 # pause time at home position before being ready to respond again
                               
                # time-based control logic:
                if self.going_home:
                    self.home_time = self.home_time - (1/samplerate) # decrement 'home_time'
                    if self.home_time <= 0: # end of fall recovery period
                        self.going_home = False
                        self.standing = True
                        rospy.loginfo("Finished getting up after fall.")
                    else:
                        if self.home_time > (pause_at_home_time + return_time): # during reflex response (legs retracted, arms back)
                            reflex_return = self.reflex        
                        else:
                            if self.home_time > pause_at_home_time: # returning to home position
                                reflex_return1 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[0]
                                reflex_return2 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[1]
                                reflex_return3 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[2]
                                reflex_return4 = ((self.home_time - pause_at_home_time)/return_time)*self.reflex[3]
                                reflex_return = [reflex_return1, reflex_return2, reflex_return3, reflex_return4]                             
                
                # set total time to get up:
                if self.responding:
                    self.responding = False
                    self.going_home = True
                    self.home_time = squat_time + return_time + pause_at_home_time
                    cmd=[105,915,710,310,50,970,250,775] # squat position with arms back
                    home=[0.3927,-0.3927,-0.5064,0.5064,1.7264,-1.7264,1.1,-1.1]
                    reflex = self.tobe.convert_motor_positions_to_angles(sag_angle_ids,cmd)-home
                    self.reflex = [abs(reflex[1]), abs(reflex[3]), abs(reflex[4]), abs(reflex[6])]
                    reflex_return = self.reflex
                    rospy.loginfo("Responding to a fall. User has %s seconds to stand TOBE up...", squat_time)

                
            # compute appropriate joint commands and execute commands: 
            response_in_radians=func.get_angles(self.lean,self.l_deriv,self.l_ddot,reflex_return,not self.standing)        
            self.response=self.tobe.convert_angles_to_commands(sag_angle_ids,response_in_radians)
            self.tobe.command_sag_motors(self.response) # send 10-bit joint commands to motors
            self.tobe.publish_sag_cmds(response_in_radians) # send joint angle commands to publisher topics  
            r.sleep()
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

