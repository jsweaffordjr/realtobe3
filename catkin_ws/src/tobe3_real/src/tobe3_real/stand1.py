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


# 2. Class for implementing learned policy
# NOTE: NN structures for critic and actor networks must be the same as the policy being loaded!
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
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

class Stand:
    """
    Class for making Tobe stand
    """
# 3. Initialize variables, publishers, subscribers
    def __init__(self,tobe):
        self.tobe=tobe
    
        # set up 'calib' subscriber for pause in Start function until calibration is finished
        self.calib=False
        #rospy.sleep(3)
        #self.sub_cal=rospy.Subscriber('/calibration',Vector3,self._calibrate, queue_size=1)
    
        # initialize standing thread:
        self._th_stand = None

        # variables, parameters:
        self.dt=0.18 #np.round(1.0/6.0, decimals=3)
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
        self.q0_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q1_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q2_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q3_next=[0,0,0,0,0,0,0,0,0,0,0,0]
        
        # reset arrays for joint angles (used to reset differentiators when joint angle values don't change within 5 timesteps):
        self.q0_last1=[0.05, -0.05, -0.6, 0.6, 1.75, -1.75, 1.1, -1.1, 0.05, -0.05, 0.0, 0.0]
        self.q0_last2=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q0_last3=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.q0_last4=[0,0,0,0,0,0,0,0,0,0,0,0]

        # thresholds:
        self.lean_min = 0.02 # lean threshold: values less than this (~1.15 deg.) are ignored
        self.az_min = 0.1 # push threshold: values less than this are ignored 
        
        # subscribers and publishers:
        #self._sub_quat = rospy.Subscriber("/lean", Vector3, self._update_orientation, queue_size=5) # subscribe to lean topic
        #self.leandata1 = rospy.Publisher('leandata1', Vector3, queue_size=1)
        #self.leandata2 = rospy.Publisher('leandata2', Vector3, queue_size=1)
        
        # load RL policy (high-level control):
        model_dir = "/home/jerry/cleanrl/models/" # CHANGE THIS TO YOUR MODEL DIRECTORY!
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("++++++CUDA is not available.++++++")
        self.policy = Policy().to(device)
        #self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/1220_run2/agent_ep_4857.pt")) # this one works
        self.policy.load_state_dict(torch.load(f"/home/jerry/cleanrl/models/Humanoid-v3__9-30-1531-58/agent_42910.pt")) # CHANGE THIS TO YOUR MODEL!
        self.policy.eval()
        
        # switch to 'active' status, move robot to initial 'home' configuration, if not already there:
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

    def _calibrate(self,msg):
        x=msg.x
        y=msg.y
        z=msg.z
        if x > 2 and y > 2 and z > 2:
            self.calib=True

# 4. Initialization functions    
    def start(self):
        if not self.active:
            self.active = True
            self.tobe.turn_on_motors()
            self.init_stand()
            
            # pause until 10 seconds after IMU is calibrated:
            #while self.calib == False:
            #    rospy.sleep(0.1)
            #rospy.sleep(10) 
            
            # start standing control loop
            self._th_stand = Thread(target=self._do_stand)
            self._th_stand.start()
            self.standing = True
                                           
    def init_stand(self):
        """
        If not already there yet, go to initial standing position
        """
        rospy.loginfo("Going to initial stance")
        #cmd=[281,741,272,750,409,613,511,511,543,480,609,414,172,851,295,728,547,476]
        cmd=[166,857,273,750,410,613,511,511,521,501,629,394,170,853,297,726,521,501]
        self.tobe.command_all_motors(cmd)

        # go to initial leg configuration
        #angle_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        #init_leg_config =[-0.2,0.2,-0.35,-0.35,0.52,-0.52,0,0,0.05, -0.05, -0.6, 0.6, 1.75, -1.75, 1.1, -1.1, 0.05, -0.05]
        #cmd2 = self.tobe.convert_angles_to_commands(angle_ids,init_leg_config)
        #self.tobe.command_leg_motors(cmd2)
        #rospy.loginfo("Initial leg configuration: ")
        #rospy.loginfo(cmd2)
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
        leg_angs = self.tobe.convert_motor_positions_to_angles(leg_angle_ids,self.tobe.read_leg_motor_positions())
                  
        # read joint motor positions, get torso angle:
        #joint_angs = np.array(leg_angs)   
        #torso_angs = np.array([self.fore_data.x, self.side_data.x])       
        #self.q0_next = np.concatenate((joint_angs, torso_angs))
        rospy.loginfo("Now at initial stance position.")
    
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
        dt=self.dt # approximate dt between data points
        samplerate = np.round(1.0/dt, 3)
        r = rospy.Rate(samplerate)
        rospy.loginfo("Started standing thread")

        policy = self.policy
        leg_angle_ids = [9,10,11,12,13,14,15,16,17,18]
               
        elapsed_time = 0

        lower_bounds = np.array([-0.35,-0.45,-1.2, 0.0, 1.25,-2.25, 0.6,-1.6,-0.2,-0.3]) # for tobe6.xml 
        upper_bounds = np.array([ 0.45, 0.35, 0.0, 1.2, 2.25,-1.25, 1.6,-0.6, 0.3, 0.2]) # for tobe6.xml 

        lower_bounds6 = np.array([-1.2, 0.0, 1.25,-2.25, 0.6,-1.6]) # for tobe6.xml 
        upper_bounds6 = np.array([ 0.0, 1.2, 2.25,-1.25, 1.6,-0.6]) # for tobe6.xml 

        # order of lower, upper bounds is based on Mujoco sim: 
        # [Rhip_lat, Rhip_sw, Rknee, Rankle_sw, Rankle_lat, Lhip_lat, Lhip_sw, Lknee, Lankle_sw, Lankle_lat]
        
        t = 0.0 
        i = 0
        testing_loop_read_speed = True
        t_start = 0.0
        t_end = 1.0
        while not rospy.is_shutdown(): 
            #read_angs = 0 # no joints read
            #read_angs = self.tobe.read_right_sag_arm_motor_positions() # 2 joints read
            #read_angs = self.tobe.read_right_arm_motor_positions() # 3 joints read
            #read_angs = self.tobe.read_arm_motor_positions() # 6 joints read
            #read_angs = self.tobe.read_leg_motor_positions() # 10 joints read
            read_angs = self.tobe.read_all_motor_positions() # 18 joints read
            i += 1
            
            if i == 11:
                t_start = rospy.get_time()
                rospy.loginfo("Start time: %f", t_start) 
            
            if i == 110:
                t_end = rospy.get_time()
                rospy.loginfo("End time: %f", t_end)
                t_elapsed = t_end - t_start
                rospy.loginfo("Time elapsed: %f", t_elapsed) 
                loop_avg = t_elapsed/100.0
                rospy.loginfo("Avg. loop time: %f", loop_avg)
            
            #rospy.loginfo(read_angs)
            
            if testing_loop_read_speed is False:
                # read joint motor positions:
                leg_angs = self.tobe.convert_motor_positions_to_angles(leg_angle_ids,self.tobe.read_leg_motor_positions())
                self.tobe.publish_leg_angs(leg_angs) # publish leg joint angles
            
                # read joint motor positions, get torso angle:
                joint_angs = np.array(leg_angs) # - np.array(refs_tobe), decimals=3) #   
                torso_angs = np.array([self.fore_data.x, self.side_data.x])       
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
            
                lf, rf = self.foot_pos(joint_angs) # use kinematics to get 3D positions of left, right feet w.r.t. torso
            
                nq = 6 # number of controlled joints
                #state = np.concatenate((q[0:10], y[0:10], q[10:], y[10:], lf, rf))
                #state = np.concatenate((q[0:nq], y[0:nq], q[10:], y[10:], lf, rf))
                state = np.concatenate((q[0:nq], y[0:nq], [0.0, 0.0], [0.0, 0.0], lf, rf))
                state[2*nq] += (0.0) # initial offset for torso pitch angle
                state1 = np.round(state, decimals=3)
            
                # update state vector
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                obs_array = np.array(state1) # create observation array
                obs = torch.Tensor(obs_array).to(device) # convert observation array to Tensor
            
                # compute appropriate joint commands and execute commands:             
                response1 = policy.deterministic_action(obs) # use RL policy to get 10 x 1 action output
                #ctrl = policy_to_cmd1(response1,lower_bounds,upper_bounds) # convert output of RL policy to joint angle command in radians
            
                if nq == 6:
                    c = policy_to_cmd1(response1,lower_bounds6,upper_bounds6)
                    ctrl = [0.05, -0.05, c[0], c[1], c[2], c[3], c[4], c[5], 0.05, -0.05]
                else:
                    ctrl = policy_to_cmd1(response1,lower_bounds,upper_bounds)
                         
                response=self.tobe.convert_angles_to_commands(leg_angle_ids,ctrl) # convert 10 x 1 action vector to 10-bit joint commands
            
                # on-screen feedback:
                rospy.loginfo(state1) 
                rospy.loginfo(ctrl)
            
                # send commands to motors and record command angle (in radians)
                self.tobe.command_leg_motors(response) # send 10-bit joint commands to motors
                self.tobe.publish_leg_cmds(ctrl) # send joint angle commands to publisher topics 
            
                t += dt 
                r.sleep()
                
        rospy.loginfo("Finished standing control thread")
	
        self._th_walk = None

