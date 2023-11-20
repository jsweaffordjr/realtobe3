#!/usr/bin/python3

from threading import Thread
import os
import time
import datetime
import signal
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
from gazebo_msgs.msg import ModelStates
from tobe3_real.tobesim import Tobe
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta


# 1. Useful functions

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
    bp = [ p.detach().cpu().numpy() for p in beta_policy]
    x = array(bp)
    y = np.round(a + np.multiply(x,d), decimals=3)
    
    return y 

def policy_to_cmd2(normal_policy,lower_bounds,upper_bounds,jp,dt):
    """
    This function converts a Normal policy with velocity commands to joint position commands to be used by the robot
    """
    # normal_policy is a vector of values roughly between -3 and 3
    # lower_bounds is a vector of the lower bounds of the joint position vector 
    # upper_bounds is a vector of the upper bounds of the joint position vector
    # jp is the current joint position vector
    # NOTE: assuming that all inputs are of the same length

    policy = [ p.detach().cpu().numpy() for p in normal_policy]
    x = array(policy)
    
    cmd_vel = x
    for j in range(len(cmd_vel)):
        if x[j] < -np.pi:
            cmd_vel[j] = -np.pi
        elif x[j] > np.pi:
            cmd_vel[j] = np.pi
    
    # Euler integration of command velocities, to generate command angles within joint command limits:        
    des = jp + cmd_vel*dt # integrate joint velocities

    # Apply joint position limits:
    a1 = array(lower_bounds)
    b1 = array(upper_bounds)
    for j in range(len(des)):
        if des[j] < a1[j]:
            des[j] = a1[j] 
        if des[j] > b1[j]:
            des[j] = b1[j] 
    
    return des                
         

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
    
    
# 2. Classes for implementing learned policy
class BetaPolicy(nn.Module):
    def __init__(self):
        super(BetaPolicy, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(22, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        
        self.actor_beta = nn.Sequential(
            layer_init(nn.Linear(22, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 6), std=0.01),            
            nn.Softplus() # lower bound of zero for output
        )
        self.actor_alpha = nn.Sequential(
            layer_init(nn.Linear(22, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 6), std=0.01),            
            nn.Softplus() # lower bound of zero for output
        )
        
    def deterministic_action(self, x):
        beta = torch.add(self.actor_beta(x),1)
        alpha = torch.add(self.actor_alpha(x),1) # now, values >= 1
        #action = torch.div(alpha,torch.add(alpha,beta)) # mean
        mode_numerator = torch.add(alpha,-1)
        mode_denominator = torch.add(torch.add(alpha,beta),-2)
        action = torch.div(mode_numerator, mode_denominator) # beta distribution mode used for "deterministic" actions
        return action  
   
class NormalPolicy(nn.Module):
    def __init__(self):
        super(NormalPolicy, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(30, 512)), # initializes layer with std_dev of sqrt(2)
            nn.ReLU(),
            layer_init(nn.Linear(512, 128)), # initializes layer with std_dev of sqrt(2)
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0), # initializes layer with 1.0
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(30, 512), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(512, 128), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(128, 10), std=0.01), # initializes layer with std_dev of 0.01, instead of 1.0
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, 10)) 
        
    def deterministic_action(self, x):
        action = self.actor_mean(x)
        return action
                              
class Stand:
    """
    Class for making Tobe stand
    """
# 3. Initialize variables, publishers, subscribers
    def __init__(self,tobe):
        self.tobe=tobe
        
        # initialization parameters:
        self._th_stand = None
        self.init_time = datetime.datetime.now() # get initial time in seconds, nanoseconds
        self.fallen = False

        # variables, parameters:
        self.dt=0.18
        self.active = False
        self.fore_angle=0.0
        self.side_angle=0.0
        self.lean_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.latest_force = 0.0 # most recently applied disturbing force
        
        # reset arrays for joint angles:
        self.q0_last1=[0,0,0,0,0,0,0,0,0,0,0,0]

        # subscribers and publishers:
        self._sub_quat = rospy.Subscriber("/simlean", Vector3, self._update_orientation, queue_size=5) # subscribe to simlean topic
        self.simlean1 = rospy.Publisher('simlean1', Float64, queue_size=1)
        self.simlean2 = rospy.Publisher('simlean2', Float64, queue_size=1) 
        self._sub_disturbance = rospy.Subscriber('/applied_force', Float64, self._update_datafile)       
        self._sub_state_feedback = rospy.Subscriber('/gazebo/model_states', ModelStates, self._get_state_callback)
        
        # load RL policy (high-level control):
        model_dir = "/home/jerry/cleanrl/models/" # CHANGE THIS TO YOUR MODEL DIRECTORY!
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("++++++CUDA is not available.++++++")
        
        self.beta = True
        if self.beta:
            self.policy = BetaPolicy().to(device)
        else:
            self.policy = NormalPolicy().to(device)
        
        # CHANGE THE NEXT TWO LINES FOR YOUR MODEL!
        self.agent_folder = "9-30-1531-58" 
        self.agent_name = "42910"
        
        try:
            os.mkdir("/home/jerry/cleanrl/gazebo_tests/"+self.agent_folder)
        except:
            pass
            
        try:
            os.mkdir("/home/jerry/cleanrl/gazebo_tests/"+self.agent_folder+"/agent_"+self.agent_name)
        except:
            pass

        # initialize data logging:
        results_dir = "/home/jerry/cleanrl/gazebo_tests/"+self.agent_folder+"/agent_"+self.agent_name+"/"
        run_date = datetime.datetime.now()
        if run_date.minute < 10:
            minute_str = "0" + str(run_date.minute) # corrects minute values under 10, e.g., outputs 1209, instead of 129 for 12:09 
        else:
            minute_str = str(run_date.minute) 
        run_time = str(run_date.month) + "-" + str(run_date.day) + "-" + str(run_date.hour) + minute_str
        datafile_name = f"test__{run_time}"
        datapath = os.path.join(results_dir, datafile_name)
        suffix = ".txt"
        self.filename = datapath+suffix # path to file in which to save data
        
        # load policy for use in simulation:
        self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/Humanoid-v3__"+self.agent_folder+"/agent_"+self.agent_name+".pt")) 
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

        qz = math.asin(-fore) # convert lean factor to lean angle (inverse sine of z-component of IMU x-axis [which is a unit vector])   
        qy = math.asin(-side) # convert lean factor to lean angle (inverse sine of y-component of IMU x-axis [which is a unit vector]) 
        
        # assign lean angle and derivative(s) and/or integral:
        self.fore_angle=qz
        self.simlean1.publish(self.fore_angle)
        self.side_angle=qy
        self.simlean2.publish(self.side_angle)
        
    def _update_datafile(self,msg):
        self.latest_force = msg.data
        current_time = round((datetime.datetime.now() - self.init_time).total_seconds(), 1)
        print(f"Applying {self.latest_force}-N force with 0.1-sec. duration at t = {current_time} sec.\n", file=open(self.filename, 'a'))

    def _get_state_callback(self,msg):
        tobe_pose = msg.pose[1] # TOBE robot is second item in environment, so it's the second item in pose list
        tobe_height = tobe_pose.position.z
        if tobe_height < 0.2: # when torso height is below 0.2 m, assume fall has occurred
            self.fallen = True
            
# 4. Initialization and other functions 
    def start(self):
        if not self.active:
            self.active = True
            self.init_stand()
            # start standing control loop
            rospy.sleep(1) # wait 1 second, then start
            self._th_stand = Thread(target=self._do_stand)
            self._th_stand.start()
            self.standing = True

    def init_stand(self):
        """
        If not already there yet, go to initial standing position
        """
        rospy.loginfo("Going to initial stance in Gazebo.")
        #cmd=[0.39,-0.39,-0.35,-0.35,0.52,-0.52,0,0,0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18]
        #cmd=[0.39,-0.39,-0.35,-0.35,0.52,-0.52,0,0,0.17,-0.17,-0.5,0.5,1.75,-1.75,1.12,-1.12,0.17,-0.17]
        cmd=[-0.2,0.2,-0.35,-0.35,0.52,-0.52,0,0,0.05, -0.05, -0.6, 0.6, 1.75, -1.75, 1.1, -1.1, 0.05, -0.05]
        self.tobe.command_all_motors(cmd)
        rospy.loginfo("Now at initial stance position in Gazebo.")

    def Amat(self,num, joint_angle, joint_offset, link_length, twist_angle):
        # this function computes a transformation matrix between two successive frames, based on D-H parameters
        # if num == 0, use alternate order of individual transformations: twist angle, link length, joint offset, joint angle
        # otherwise, use conventional order: joint angle, joint offset, link length, twist angle
        # note: angles are in radians
        d = joint_offset
        a = link_length
        
        CT = np.cos(joint_angle)
        ST = np.sin(joint_angle)
        CA = np.cos(twist_angle)
        SA = np.sin(twist_angle)
        
        if num == 0: # use alternate order: twist angle, then link length, then joint offset, then joint angle
            Amatrix = [[CT, -ST, 0, a],
                      [ST*CA, CT*CA, -SA, -d*SA],
                      [ST*SA, CT*SA, CA, d*CA],
                      [0,0,0,1]]

        else: # otherwise, use conventional order: joint angle, then joint offset, then link length, then twist angle
            Amatrix = [[CT, -ST*CA, ST*SA, a*CT],
                      [ST, CT*CA, -CT*SA, a*ST],
                      [0, SA, CA, d],
                      [0,0,0,1]]
        
        Amat = np.array(Amatrix)     
        return Amat
        
    def foot_pos(self,lower_body_joint_positions):
        # assuming input of ten lower-body joint positions...
        jp1 = lower_body_joint_positions
        jp = [jp1[9],jp1[7],jp1[5],jp1[3],jp1[1],jp1[0],jp1[2],jp1[4],jp1[6],jp1[8]]
        
        right_hip = self.Amat(1,0,-47,38.5,0).dot(self.Amat(1,0,0,0,-np.pi/2))
        right_knee = right_hip.dot(self.Amat(1,-jp[5]+np.pi/2,0,70,np.pi/2))
        next_joint = right_knee.dot(self.Amat(1,-jp[6]-(13.614*np.pi/180),0,66,0))
        right_ankle = next_joint.dot(self.Amat(1,-jp[7]+(26.294*np.pi/180),-15,62,0))
        joint_after = right_ankle.dot(self.Amat(1,jp[8]-(12.680*np.pi/180),0,45,-np.pi/2))
        right_foot = joint_after.dot(self.Amat(1,jp[9]+np.pi,0,-30,0))
        right_midfoot = right_foot.dot(self.Amat(1,0,25,0,0))
        
        left_hip = self.Amat(0,0,-47,-38.5,0).dot(self.Amat(0,jp[4]+np.pi/2,0,0,np.pi/2))
        left_knee = left_hip.dot(self.Amat(0,-jp[3]+(13.614*np.pi/180),0,-70,-np.pi/2))
        left_ankle = left_knee.dot(self.Amat(0,-jp[2]-(26.294*np.pi/180),0,-66,0))
        next_joint = left_ankle.dot(self.Amat(0,jp[1]+(12.680*np.pi/180),-15,-62,0))
        left_foot = next_joint.dot(self.Amat(0,jp[0],0,-45,-np.pi/2))
        joint_after = left_foot.dot(self.Amat(0,0,0,-30,0))
        left_midfoot = joint_after.dot(self.Amat(0,0,25,0,0))
        
        rf = right_midfoot[:3,3]
        lf = left_midfoot[:3,3] 
        #print(f"Joints: {jp}")
        #print(f"Left midfoot: {lf}")
        #print(f"Right midfoot: {rf}") 
        return lf,rf 

# 5. Standing control loop
    def _do_stand(self):
        """
        Main standing control loop
        """
        samplerate = 1.0/self.dt
        dt=self.dt # approximate dt between data points
        
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread in Gazebo")
        arm_angle_ids = [1,2,3,4,5,6]
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        
        policy = self.policy
        
        #refs_tobe = [0.17,-0.17,-0.5,0.5,1.75,-1.75,1.12,-1.12,0.17,-0.17] # in Gazebo
        #refs_tobe = [0.16,-0.16,-0.5,0.5,1.74,-1.74,1.11,-1.11,0.18,-0.18] # in Gazebo
        #lower_bounds = [-0.3,-0.3,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.3,-0.3] # in MuJoCo
        #upper_bounds = [ 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3] # in MuJoCo
        lower_bounds = np.array([-0.35,-0.45,-1.2, 0.0, 1.25,-2.25, 0.6,-1.6,-0.2,-0.3]) # for tobe6.xml 
        upper_bounds = np.array([ 0.45, 0.35, 0.0, 1.2, 2.25,-1.25, 1.6,-0.6, 0.3, 0.2]) # for tobe6.xml 
        
        lower_bounds6 = np.array([-1.2, 0.0, 1.25,-2.25, 0.6,-1.6]) # for tobe6.xml 
        upper_bounds6 = np.array([ 0.0, 1.2, 2.25,-1.25, 1.6,-0.6]) # for tobe6.xml 

        #references1 = [-0.2,0.45,-1.7264,-1.1932,-0.21,0.2,-0.45,1.7264,1.1932,0.21]

        
        t = 0.0    
        while not (rospy.is_shutdown() or self.fallen): 
            # read joint motor positions:
            pos = np.round(np.array(self.tobe.read_leg_angles()), decimals=3) #
            
            # round to nearest 0.005 rad.
            pos_next = 1000.0*pos 
            pos_thousand = pos_next.astype(int)
            pos_after = 5*np.round(0.2*pos_thousand, 0)
            pos_nearestfive = pos_after.astype(int)
            joint_angs = np.round(pos_nearestfive*0.001, 3)            
            torso_angs = np.array([self.fore_angle, self.side_angle])       
            q = np.concatenate((joint_angs, torso_angs))
            
            # check for repeating joint angle/torso angle values:           
            if t == 0.0:
                vels = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            else:
                vels = (q - self.q0_last1)*samplerate
            
            self.q0_last1=q
            y = vels
            
            lf, rf = self.foot_pos(joint_angs) # use kinematics to get 3D positions of left, right feet w.r.t. torso
            nq = 6 # number of controlled joints
            #state = np.concatenate((q[0:10], y[0:10], q[10:], y[10:], lf, rf))
            state = np.concatenate((q[0:nq], y[0:nq], q[10:], y[10:], lf, rf))
            state[2*nq] += (0.07) # initial offset for torso pitch angle
            state1 = np.round(state, decimals=3)
            
            # update state vector
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obs_array = np.array(state1) # create observation array
            obs = torch.Tensor(obs_array).to(device) # convert observation array to Tensor
            
            # compute appropriate joint commands and execute commands: 
            response = policy.deterministic_action(obs) # use RL policy to get nq x 1 deterministic policy action output           
            
            if self.beta:
                if nq == 6:
                    c = policy_to_cmd1(response,lower_bounds6,upper_bounds6)
                    ctrl = [0.05, -0.05, c[0], c[1], c[2], c[3], c[4], c[5], 0.05, -0.05]
                else:
                    ctrl = policy_to_cmd1(response,lower_bounds,upper_bounds)
            else:
                ctrl = policy_to_cmd2(response,lower_bounds,upper_bounds,joint_angs,dt)
            self.tobe.command_leg_motors(ctrl)
            
            #rospy.loginfo(ctrl)
            #rospy.loginfo(torso_angs)              

            t += dt
            #print(f"Time: {t}")
            r.sleep()
        if self.fallen:
            print(f"Fell after {self.latest_force}-N force was applied. \n", file=open(self.filename, 'a'))
            #os.system("killall -q gzclient & killall -q gzserver")
            
        rospy.loginfo("Finished standing control thread")
        
	
        self._th_walk = None

