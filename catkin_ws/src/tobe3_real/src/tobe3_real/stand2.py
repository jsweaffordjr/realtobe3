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
from tobe3_real.tobesim import Tobe
from torch.distributions.beta import Beta

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
def policy_to_cmd(beta_policy,lower_bounds,upper_bounds):
    """
    This function converts a Beta policy to the corresponding input to be used by the robot
    """
    # beta_policy is a vector of values between 0 and 1
    # lower_bounds is a vector of the lower bounds of the input (corresponding to 0 from the policy)
    # upper_bounds is a vector of the upper bounds of the input (corresponding to 1 from the policy)
    # NOTE: assuming that all inputs are of the same length
    a = array(lower_bounds)
    b = array(upper_bounds)
    d = b-a
    bp = [ p.detach().numpy() for p in beta_policy]
    x = array(bp)
    y = a + np.multiply(x,d)
    
    # output order is based on Mujoco sim: 
    # [Rhip_lat, Rhip_sw, Rknee, Rankle_sw, Rankle_lat, Lhip_lat, Lhip_sw, Lknee, Lankle_sw, Lankle_lat],
    # whereas order for TOBE is:
    # [Rhip_lat, Lhip_lat, Rhip_sw, Lhip_sw, Rknee, Lknee, Rankle_sw, Lankle_sw, Rankle_lat, Lankle_lat]
    
    # rearrange values for use in TOBE
    z = [y[0],y[5],y[1],y[6],y[2],y[7],y[3],y[8],y[4],y[9]]
    
    return z 

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
    
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(24, 100)),
            nn.ReLU(),
            layer_init(nn.Linear(100, 50)),
            nn.ReLU(),
            layer_init(nn.Linear(50, 25)),
            nn.ReLU(),
            layer_init(nn.Linear(25, 1), std=1.0),
        )
        #self.actor_alpha_and_beta = nn.Sequential(
        #    layer_init(nn.Linear(24, 100)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(100, 50)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(50, 25)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(25, 20), std=0.01),
        #    nn.Softplus(), # lower bound of zero for output
        #)
        self.actor_beta = nn.Sequential(
            layer_init(nn.Linear(24, 100)),
            nn.Tanh(),
            layer_init(nn.Linear(100, 50)),
            nn.Tanh(),
            layer_init(nn.Linear(50, 25)),
            nn.Tanh(),
            layer_init(nn.Linear(25, 10), std=0.01),
            nn.Softplus(), # lower bound of zero for output
        )
        self.actor_alpha = nn.Parameter(torch.zeros(1, 10))

    def get_angles(self, x):
        action_beta = torch.add(self.actor_beta(x),2)
        #action_beta = torch.add(self.actor_beta(x),2) # now, beta values >= 2
        #action_logalpha = self.actor_alpha.squeeze()
        action_alpha = torch.add(self.actor_alpha.squeeze(),2) # now, alpha values >= 2
        #alphas_and_betas = torch.add(self.actor_alpha_and_beta(x),2) # now, alpha, beta values >= 1
        #action_alpha, action_beta = torch.tensor_split(alphas_and_betas, 2, dim=0)
        #probs = Beta(action_alpha, action_beta)
        #action = probs.sample() # sample from Beta dist. will be in [0,1] range
        #action = torch.div(torch.add(action_alpha, -0.33333),torch.add(action_alpha,torch.add(action_beta,-0.66666))) #median
        action = torch.div(torch.add(action_alpha, -1.0),torch.add(action_alpha,torch.add(action_beta,-2.0))) #mode
        #action = torch.div(action_alpha,torch.add(action_alpha,action_beta)) # mean
        
        return action    

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
                              

class Stand:
    """
    Class for making Tobe stand
    """

    def __init__(self,tobe):
        self.func = StandFunc()
        self.tobe=tobe
        
        # initialization parameters:
        self.ready_pos = self.func.init_angles
        self._th_stand = None

        # variables, parameters:
        self.dt=0.05
        self.active = False
        self.push=0 # smoothed z-acceleration value
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
        
        self.lean_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.az_min = 0.1 # push threshold: values less than this are ignored

        # subscribers and publishers (from realtobe):
        self._sub_quat = rospy.Subscriber("/lean", Vector3, self._update_orientation, queue_size=5) # subscribe to lean topic
        self.simlean1 = rospy.Publisher('simlean1', Vector3, queue_size=1)
        self.simlean2 = rospy.Publisher('simlean2', Vector3, queue_size=1)
        self.app_force = rospy.Publisher('rand_force', Float64, queue_size=1)
        
        # load RL policy (high-level control):
        model_dir = "/home/jerry/cleanrl/models/" # CHANGE THIS TO YOUR MODEL DIRECTORY!
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = Policy().to(device)
        #self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/0812_run1/agent_ep_200.pt", map_location=device))
        self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/1208_run3/agent_ep_6662.pt")) # CHANGE THIS TO YOUR MODEL!
        self.policy.eval()
        
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
        qz = math.asin(fore) # convert lean factor to lean angle (inverse sine of z-component of IMU x-axis [which is a unit vector])   
        qy = math.asin(side) # convert lean factor to lean angle (inverse sine of y-component of IMU x-axis [which is a unit vector]) 
        
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
        self.f_deriv = z0dot[0]
        self.s_deriv = z0dot[1]
        self.f_ddot = z1dot[0]
        self.s_ddot = z1dot[1]
        
        # assign lean angle and derivative(s) and/or integral:
        self.fore_data.x=qz
        self.fore_data.y=self.f_deriv
        self.fore_data.z=self.f_ddot
        self.simlean1.publish(self.fore_data)
        self.side_data.x=qy
        self.side_data.y=self.s_deriv
        self.side_data.z=self.s_ddot
        self.simlean2.publish(self.side_data)

    def start(self):
        if not self.active:
            self.active = True
            self.init_stand()
            # start standing control loop
            rospy.sleep(5) # wait 5 seconds, then start
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
        samplerate = 10
        dt=np.round(1.0/samplerate, decimals=3)
        self.dt=dt # approximate dt between data points
        
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread, dt = %d", dt)
        arm_angle_ids = [1,2,3,4,5,6]
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        
        t = 0.0       
        while not rospy.is_shutdown(): 
            # read joint motor positions:
            q = self.tobe.read_leg_angles()          
                                         
            # compute appropriate joint commands and execute commands: 
            #rospy.loginfo("Fore lean, vel: %d, %d", self.fore_data.x, self.fore_data.y) 
            #rospy.loginfo("Side lean, vel: %d, %d", self.side_data.x, self.side_data.y) 
            
            init_pos = [0.39,-0.35,0.52,-0.39,-0.35,-0.52]
            init_cmd = [281,741,272,750,409,613]

            
            # check for repeating joint angle values:
            threshold = 0.05
            for i in range(10): 
                if (abs(q[i] - self.q0_last1[i])+abs(q[i] - self.q0_last2[i])+abs(q[i] - self.q0_last3[i])+abs(q[i] - self.q0_last4[i])) <= threshold:
                    # reset differentiator if past four values are very near the current joint angle value:
                    self.q0_next[i] = q[i]
                    self.q1_next[i] = 0.0
                    self.q2_next[i] = 0.0
                    self.q3_next[i] = 0.0 
            
            self.q0_last1=q
            self.q0_last2=self.q0_last1
            self.q0_last3=self.q0_last2
            self.q0_last4=self.q0_last3
      
            q0 = self.q0_next
            q1 = self.q1_next
            q2 = self.q2_next
            q3 = self.q3_next
            [joint_vels, q1dot,self.q0_next,self.q1_next,self.q2_next,self.q3_next] = HOSM_diff(dt, q, q0, q1, q2, q3)
            #self.tobe.publish_leg_ang_vels(joint_vels) # publish leg joint angular velocities

            z = leg_angs
            y = joint_vels
            state1 = [z[0],z[2],z[4],z[6],z[8],z[1],z[3],z[5],z[7],z[9],y[0],y[2],y[4],y[6],y[8],y[1],y[3],y[5],y[7],y[9],self.fore_data.x, self.side_data.x,self.fore_data.y, self.side_data.y] 
            # update state vector
            obs_array = np.array(state1) # create observation array
            obs = torch.Tensor(obs_array) # convert observation array to Tensor
            
            # compute appropriate joint commands and execute commands: 
            response = policy.get_angles(obs) # use RL policy to get 10 x 1 action output
            response_in_radians = policy_to_cmd(response,references) # convert output of RL policy to joint angle command in radians 
            
            
            #arm_joints=self.tobe.convert_angles_to_commands(arm_angle_ids,arm_angles) # convert to 10-bit cmds
            arm_joints = self.tobe.convert_motor_positions_to_angles(arm_angle_ids,arm_angles)
            self.tobe.command_arm_motors(arm_angles)  
            self.tobe.publish_arm_cmds(arm_joints)
            rospy.loginfo(arm_joints)  
            
            
             
            t += dt
            r.sleep()
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

