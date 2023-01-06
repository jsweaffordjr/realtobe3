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
    
def policy_to_cmd(beta_policy,references):
    """
    This function converts a Beta policy to the corresponding input to be used by the robot
    """
    # beta_policy is a vector of values between 0 and 1
    # lower_bounds is a vector of the lower bounds of the input (corresponding to 0 from the policy)
    # upper_bounds is a vector of the upper bounds of the input (corresponding to 1 from the policy)
    # NOTE: assuming that all inputs are of the same length
    a = array(references)
    b = [ p.detach().numpy() for p in beta_policy]
    x = array(b)
    y = -a - np.multiply(x,1.0) + 0.5
    
    # output order is based on Mujoco sim: 
    # [Rhip_lat, Rhip_sw, Rknee, Rankle_sw, Rankle_lat, Lhip_lat, Lhip_sw, Lknee, Lankle_sw, Lankle_lat],
    # whereas order for TOBE is:
    # [Rhip_lat, Lhip_lat, Rhip_sw, Lhip_sw, Rknee, Lknee, Rankle_sw, Lankle_sw, Rankle_lat, Lankle_lat]
    
    # rearrange values for use in TOBE
    z = [y[0],y[5],y[1],y[6],y[2],y[7],y[3],y[8],y[4],y[9]]
    
    return z  
      
# HOSM diff option 1: if only 1 derivative (z0dot) is needed, use this:
    #z0=self.z_next[0]
    #z1=self.z_next[1]
    #z2=self.z_next[2]
    #z0dot=z1-2.12*(L**(1/3))*(abs(z0-qz)**(2/3))*np.sign(z0-qz)
    #z1dot=z2-2*(L**(2/3))*(abs(z0-qz)**(1/3))*np.sign(z0-qz)
    #z2dot=-1.1*L*np.sign(z0-qz)
    #self.z_next[0]=z0+dt*z0dot+0.5*dt*dt*z2
    #self.z_next[1]=z1+dt*z1dot
    #self.z_next[2]=z2+dt*z2dot    
    
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


#class Policy1(nn.Module): # class for the NN trained in simulation
#    def __init__(self):
#        super(Policy1, self).__init__()
#        self.critic = nn.Sequential(
#            layer_init(nn.Linear(40, 64)),
#            nn.Tanh(),
#            layer_init(nn.Linear(64, 64)),
#            nn.Tanh(),
#            layer_init(nn.Linear(64, 1), std=1.0),
#        )
#        self.actor_alpha_and_beta = nn.Sequential(
#            layer_init(nn.Linear(40, 64)),
#            nn.Tanh(),
#            layer_init(nn.Linear(64, 64)),
#            nn.Tanh(),
#            layer_init(nn.Linear(64, 20), std=0.01),
#            nn.Softplus() # lower bound of zero for output
#        )

#    def get_angles(self, x):
#        alphas_and_betas = torch.add(self.actor_alpha_and_beta(x),1) # now, alpha, beta values >= 1
#        action_alpha, action_beta = torch.tensor_split(alphas_and_betas, 2, dim=0)
#        probs = Beta(action_alpha, action_beta)
#        action = probs.sample() # sampled value from Beta dist. will be in [0,1] range
        #action = torch.div(torch.add(action_alpha, -1.0),torch.add(action_alpha,torch.add(action_beta,-2.0))) #mode
        #action = torch.div(action_alpha,torch.add(action_alpha,action_beta)) # mean#
        #action = torch.div(torch.add(action_alpha, -0.33333),torch.add(action_alpha,torch.add(action_beta,-0.66666))) #median
#        return action             

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

class Stand:


    def __init__(self):

        # initialize standing thread:
        self._th_stand = None

        # variables, parameters:
        self.dt=0.16
        self.active = False        
        
        # load RL policy (high-level control):
        model_dir = "/home/jerry/cleanrl/models/" # CHANGE THIS TO YOUR MODEL DIRECTORY!
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = Policy().to(device)
        #self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/0812_run1/agent_ep_200.pt", map_location=device))
        self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/1205_run1/agent_ep_6994.pt")) # CHANGE THIS TO YOUR MODEL!
        self.policy.eval()
        
        # switch to 'active' status, move to initial 'home' position, if not already there:
        self.start()

    def start(self):
        if not self.active:
            self.active = True
            
            # start standing control loop
            self._th_stand = Thread(target=self._do_stand)
            self._th_stand.start()
            self.standing = True
    
    def convert_angles_to_commands(self,ids,angles):
        # this function converts an array of angle values (in radians) to the corresponding 10-bit motor values,
        # assuming that the 'ids' array contains matching ID numbers for the angle values of 'angles' array
        
        b=[60,240,60,240,150,150,150,150,150,150,150,150,150,150,150,150,150,150] # motor offsets
        c=[1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1] # motor polarities
        
        cmds=np.zeros(len(ids)) # initialize cmds array
        for j in range(len(ids)): # get 10-bit motor values:
            num=ids[j] # get motor ID number from array
            ang=angles[j] # get desired joint angle value in radians
            cmds[j]=(1023/300)*((ang*(180/math.pi)*c[num-1])+b[num-1]) # convert to degrees, apply polarity, convert to 10-bit, add offset
        return cmds
        
    def _do_stand(self):
        """
        Main standing control loop
        """
        #references = [-0.15,0.85,-1.73,-1.00,-0.2,0.15,-0.85,1.73,1.00,0.2]
        references = [-0.16,0.5,-1.74,-1.11,-0.18,0.16,-0.5,1.74,1.11,0.19]
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        typical_leg_angs = np.array([0.1612,-0.1663,-0.499,0.5195,1.7479,-1.7274,1.1132,-1.103,0.1817,-0.1971])
        
        # observation array
        policy = self.policy    
        state1 = [0.1612,-0.499,1.7479,1.1132,0.1817,-0.1663,0.5093,-1.7274,-1.103,-0.1971,0.0,0.0] # joint angle vector + first 2 from joint_vel vector
        state2 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # remainder of state
        
        # update state vector
        state = np.array([state1,state2]) # combine into state array
        obs_array = state.flatten() # flatten state into observation array
        obs = torch.Tensor(obs_array) # convert observation array to Tensor
         
        # compute appropriate joint commands and execute commands: 
        response = policy.get_angles(obs) # use RL policy to get 10 x 1 action output
        response_in_radians = policy_to_cmd(response,references) # convert output of RL policy to joint angle command in radians 
        self.response=self.convert_angles_to_commands(leg_angle_ids,response_in_radians) # convert 10 x 1 action vector to 10-bit joint commands
        diff = typical_leg_angs - response_in_radians
        sum1 = 0.0
        num = 1
        for x in diff:
            sum1 += x*x
            num += 1
        rms = np.sqrt(sum1/num)
        
        rospy.loginfo("Observation: ")
        rospy.loginfo(obs_array)
        rospy.loginfo("Angles - Commands: ")
        rospy.loginfo(diff)
        rospy.loginfo("RMS diff: ")
        rospy.loginfo(rms)

        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

