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
    
def policy_to_cmd1(beta_policy,lower_bounds,upper_bounds):
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
        action_beta = torch.add(self.actor_beta(x),2) # now, beta values >= 2
        #action_logalpha = self.actor_alpha.squeeze()
        action_alpha = torch.add(self.actor_alpha.squeeze(),2) # now, alpha values >= 2
        #alphas_and_betas = torch.add(self.actor_alpha_and_beta(x),2) # now, alpha, beta values >= 2
        #action_alpha, action_beta = torch.tensor_split(alphas_and_betas, 2, dim=0)
        #probs = Beta(action_alpha, action_beta)
        #action = probs.sample() # sample from Beta dist. will be in [0,1] range
        #action = torch.div(torch.add(action_alpha, -0.33333),torch.add(action_alpha,torch.add(action_beta,-0.66666))) #median
        #action = torch.div(torch.add(action_alpha, -1.0),torch.add(action_alpha,torch.add(action_beta,-2.0))) #mode
        action = torch.div(action_alpha,torch.add(action_alpha,action_beta)) # mean
        
        return action     
                              
class Stand:
    """
    Class for making Tobe stand
    """

    def __init__(self,tobe):
        self.tobe=tobe
        
        # initialization parameters:
        self._th_stand = None

        # variables, parameters:
        self.dt=np.round(1.0/6.0, decimals=3)
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
        self.fore_angle=0.0
        self.side_angle=0.0
        
        # differentiation arrays for torso angles:
        self.z0_next=[0,0]
        self.z1_next=[0,0]
        self.z2_next=[0,0]
        self.z3_next=[0,0]
        
        # differentiation arrays for joint angles, torso angles:
        self.q0_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q1_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q2_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q3_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        
        # reset arrays for joint angles (used to reset differentiators when joint angle values don't change within 5 timesteps):
        self.q0_last1=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q0_last2=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q0_last3=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q0_last4=[0,0,0,0,0,0,0,0,0,0,0,0]
        
        self.lean_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.az_min = 0.1 # push threshold: values less than this are ignored

        # subscribers and publishers:
        self._sub_quat = rospy.Subscriber("/simlean", Vector3, self._update_orientation, queue_size=5) # subscribe to simlean topic
        self.simlean1 = rospy.Publisher('simlean1', Float64, queue_size=1)
        self.simlean2 = rospy.Publisher('simlean2', Float64, queue_size=1)
        self.app_force = rospy.Publisher('rand_force', Float64, queue_size=1)
        
        # load RL policy (high-level control):
        model_dir = "/home/jerry/cleanrl/models/" # CHANGE THIS TO YOUR MODEL DIRECTORY!
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = Policy().to(device)
        #self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/1220_run2/agent_ep_4857.pt")) # this one works
        self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/1221_run1/agent_ep_5597.pt")) # CHANGE THIS TO YOUR MODEL!
        self.policy.eval()
        
        # switch to 'active' status, move to initial 'home' position, if not already there:
        self.start()

    def _update_orientation(self, msg):
        """
        Catches lean angle data and updates robot orientation
        """
        fore = msg.x # get x-(i.e., forward/backward direction)component of initially vertical x-axis
        side = msg.y # get y-(i.e., left/right direction)component of initially vertical x-axis
        if abs(fore) < self.lean_min: # apply threshold
            fore = 0
        if abs(side) < self.lean_min: # apply threshold
            side = 0
                
        #qz=fore # use lean factor instead of lean angle
        #qy=side # use lean factor instead of lean angle
        qz = math.asin(-fore) # convert lean factor to lean angle (inverse sine of z-component of IMU x-axis [which is a unit vector])   
        qy = math.asin(-side) # convert lean factor to lean angle (inverse sine of y-component of IMU x-axis [which is a unit vector]) 
        
        self.fore_lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.side_lean.pop(0) # remove oldest value from array of five previous lean angle values
        self.fore_lean.append(qz) # append newest value to end of the array
        self.side_lean.append(qy) # append newest value to end of the array
        
        # assign lean angle and derivative(s) and/or integral:
        self.fore_angle=qz
        self.simlean1.publish(self.fore_angle)
        self.side_angle=qy
        self.simlean2.publish(self.side_angle)

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
        rospy.loginfo("Going to initial stance in Gazebo.")
        cmd=[0.39,-0.39,-0.35,-0.35,0.52,-0.52,0,0,0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18]
        self.tobe.command_all_motors(cmd)
        rospy.loginfo("Now at initial stance position in Gazebo.")

    def _do_stand(self):
        """
        Main standing control loop
        """
        samplerate = 6
        dt=self.dt # approximate dt between data points
        
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread in Gazebo")
        arm_angle_ids = [1,2,3,4,5,6]
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        
        policy = self.policy

        refs_tobe = [0.17,-0.17,-0.5,0.5,1.75,-1.75,1.12,-1.12,0.17,-0.17] # in Gazebo
        lower_bounds = [-0.3,-0.5,-0.5,-0.5,-0.3,-0.3,-0.5,-0.5,-0.5,-0.3] # in MuJoCo
        upper_bounds = [0.3,0.5,0.5,0.5,0.3,0.3,0.5,0.5,0.5,0.3] # in MuJoCo
        
        #references1 = [-0.2,0.45,-1.7264,-1.1932,-0.21,0.2,-0.45,1.7264,1.1932,0.21]
        
        t = 0.0       
        while not rospy.is_shutdown(): 
            # read joint motor positions:
            joint_angs = np.round(np.array(self.tobe.read_leg_angles()) - np.array(refs_tobe), decimals=3) #   
            torso_angs = np.array([self.fore_angle, self.side_angle])       
            q = np.concatenate((joint_angs, torso_angs))
            
            # check for repeating joint angle/torso angle values:
            threshold = 0.05
            for i in range(12): 
                if (abs(q[i] - self.q0_last1[i])+abs(q[i] - self.q0_last2[i])+abs(q[i] - self.q0_last3[i])+abs(q[i] - self.q0_last4[i])) <= threshold:
                    # reset differentiator if past four values are very near the current joint angle value:
                    self.q0_next[i] = q[i]
                    self.q1_next[i] = 0.0
                    self.q2_next[i] = 0.0
                    self.q3_next[i] = 0.0 
            
            vels = (q - self.q0_last1)/(1.0/samplerate)
            
            self.q0_last1=q
            self.q0_last2=self.q0_last1
            self.q0_last3=self.q0_last2
            self.q0_last4=self.q0_last3
      
            q0 = self.q0_next
            q1 = self.q1_next
            q2 = self.q2_next
            q3 = self.q3_next
            [vels1, q1dot,self.q0_next,self.q1_next,self.q2_next,self.q3_next] = HOSM_diff(dt, q, q0, q1, q2, q3)

            
            y = vels
            state = [q[0],q[2],q[4],q[6],q[8],q[1],q[3],q[5],q[7],q[9],y[0],y[2],y[4],y[6],y[8],y[1],y[3],y[5],y[7],y[9],q[10],q[11],y[10], y[11]] 
            state1 = np.round(state, decimals=3)
            
            # update state vector
            obs_array = np.array(state1) # create observation array
            obs = torch.Tensor(obs_array) # convert observation array to Tensor
            
            # compute appropriate joint commands and execute commands: 
            #response = policy.get_angles(obs) # use RL policy to get 10 x 1 action output
            
            # test control inputs:
            
            if t >= 70.0:
                rospy.loginfo('Final phase: ')
                response = torch.Tensor(np.array([0.5,0.45,0.5,0.5,0.5,0.5,0.55,0.5,0.5,0.5]))
            elif t >= 65.0:
                rospy.loginfo('Phase 7: ')
                response = torch.Tensor(np.array([0.5,0.5,0.5,0.5,0.8,0.5,0.5,0.5,0.5,0.2]))
            elif t >= 60.0:
                rospy.loginfo('Phase 6: ')
                response = torch.Tensor(np.array([0.5,0.5,0.3,0.3,0.8,0.5,0.5,0.7,0.7,0.2]))
            elif t >= 55.0:
                rospy.loginfo('Phase 5: ')
                response = torch.Tensor(np.array([0.5,0.5,0.7,0.7,0.8,0.5,0.5,0.3,0.3,0.2]))
            elif t >= 50.0:
                rospy.loginfo('Phase 4: ')
                response = torch.Tensor(np.array([0.7,0.5,0.5,0.5,0.8,0.3,0.5,0.5,0.5,0.2]))
            elif t >= 45.0:
                rospy.loginfo('Phase 3: ')
                response = torch.Tensor(np.array([0.3,0.5,0.5,0.5,0.8,0.7,0.5,0.5,0.5,0.2]))
            elif t >= 40.0:
                rospy.loginfo('Phase 2: ')
                response = torch.Tensor(np.array([0.3,0.5,0.5,0.5,0.2,0.7,0.5,0.5,0.5,0.8]))
            elif t >= 35:
                rospy.loginfo('Phase 1: ')
                response = torch.Tensor(np.array([0.5,0.5,0.5,0.5,0.2,0.5,0.5,0.5,0.5,0.8]))
            else:
                response = torch.Tensor(np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]))
            
            ctrl = policy_to_cmd1(response,lower_bounds,upper_bounds)
            response_in_radians = np.array(refs_tobe) + ctrl # convert output of RL policy to joint angle command in radians             

            self.tobe.command_leg_motors(response_in_radians)
            rospy.loginfo(obs_array)              
             
            t += dt
            r.sleep()
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

