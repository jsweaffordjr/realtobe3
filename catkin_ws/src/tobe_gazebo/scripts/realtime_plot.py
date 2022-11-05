#!/usr/bin/python3
import roslib
import rospy
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

# this script subscribes to the /push and /lean topics, generates real-time plot(s)
class Visualizer:
    def __init__(self):
        rospy.sleep(5)         
        # plot initialization:
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, sharex=True)
        self.fig.set_size_inches(10, 10)
        self.a1, = self.ax1.plot([], [], label='Lean rate', color='black')
        self.a2, = self.ax1.plot([], [], label='Lean angle', color='red')
        self.a3, = self.ax2.plot([], [], label='X-direction', color='blue')
        self.a4, = self.ax3.plot([], [], label='R knee (cmd)', color='blue', linestyle='dashed')
        self.a5, = self.ax3.plot([], [], label='L hip (cmd)', color='green', linestyle='dashed')
        self.a6, = self.ax3.plot([], [], label='R shoulder (cmd)', color='red', linestyle='dashed')
        self.a7, = self.ax3.plot([], [], label='L shoulder (cmd)', color='black', linestyle='dashed')
        self.a8, = self.ax3.plot([], [], label='R hip (cmd)', color='red', linestyle='dashed')
        self.a9, = self.ax3.plot([], [], label='L knee (cmd)', color='green', linestyle='dashed')
        self.a10, = self.ax3.plot([], [], label='Right knee', color='blue')
        self.a11, = self.ax3.plot([], [], label='Left hip', color='green')
        self.a12, = self.ax3.plot([], [], label='Right shoulder', color='red')
        self.a13, = self.ax3.plot([], [], label='Left shoulder', color='black')
        self.a14, = self.ax3.plot([], [], label='Right hip', color='red')
        self.a15, = self.ax3.plot([], [], label='Left knee', color='green')

        self.t1 = 0
        self.lean = 0
        self.lean_dot = 0
        self.acc = 0
        self.l_knee = -1.7264
        self.r_knee = 1.7264
        self.r_hip = -0.4132
        self.l_hip = 0.4132
        self.r_shoulder = 0.3927 
        self.l_shoulder = -0.3927
        self.r_hip2 = self.r_hip
        self.l_hip2 = self.l_hip
        self.r_shoulder2 = self.r_shoulder
        self.l_shoulder2 = self.l_shoulder
        self.l_knee2 = self.l_knee
        self.r_knee2 = self.r_knee
        self.all_cmds = np.array([self.r_shoulder,self.l_shoulder,self.r_hip,self.l_hip,self.r_knee,self.l_knee])
        self.all_angs = self.all_cmds
        self.all_imu = np.zeros(3)

        self.plot_length = 10 # duration of the plot window
        self.t_start=0
        self.t_end=self.plot_length
        self.fig.suptitle('Real-time plots')
        
        self.initial_time = rospy.get_time()
            
    def init_plot(self):
        self.ax1.set_title('Longitudinal lean angle and rate')
        self.ax1.set_xlim(0,self.plot_length)
        self.ax1.set_ylim(-1.5,1.5)
        self.ax1.legend(loc='center left')
        self.ax2.set_title('Torso forward acceleration (m/s^2)')
        self.ax2.set_xlim(0,self.plot_length)
        self.ax2.set_ylim(-2,2)
        self.ax2.legend(loc='center left')
        self.ax3.set_title('Commanded joint angles')
        self.ax3.set_xlim(0,self.plot_length)
        self.ax3.set_ylim(-2,2)
        self.ax3.legend(loc='center left')
        return self.ax1, self.ax2, self.ax3        

    def update_plot_length(self):
        self.ax1.set_xlim(self.t_start,self.t_end)
        self.ax2.set_xlim(self.t_start,self.t_end)
        self.ax3.set_xlim(self.t_start,self.t_end)
        return self.ax1, self.ax2, self.ax3  
                
    def plot_update(self, frame):
        tnow=rospy.get_time()-self.initial_time       
        self.t1=np.append(self.t1,tnow)
        self.append_new_data()

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t_end=tnow
            self.t_start=tnow-self.plot_length
            self.delete_old_data()
            self.t1=np.delete(self.t1,0)
        
        # get IMU data:
        lean = self.all_imu[:,0]
        lean_dot = self.all_imu[:,1]
        acc = self.all_imu[:,2]
        
        # get joint cmds:
        r_shoulder = self.all_cmds[:,0]
        l_shoulder = self.all_cmds[:,1]
        r_hip = self.all_cmds[:,2]
        l_hip = self.all_cmds[:,3]
        r_knee = self.all_cmds[:,4]
        l_knee = self.all_cmds[:,5]
        
        # get joint angs:
        r_shoulder2 = self.all_angs[:,0]
        l_shoulder2 = self.all_angs[:,1]
        r_hip2 = self.all_angs[:,2]
        l_hip2 = self.all_angs[:,3]
        r_knee2 = self.all_angs[:,4]
        l_knee2 = self.all_angs[:,5]
        
        # update plot data:    
        self.a2.set_data(self.t1,lean)
        self.a1.set_data(self.t1,lean_dot)
        self.a3.set_data(self.t1,acc)
        self.a4.set_data(self.t1,r_knee)
        self.a5.set_data(self.t1,l_hip)
        self.a6.set_data(self.t1,r_shoulder)
        self.a7.set_data(self.t1,l_shoulder)
        self.a8.set_data(self.t1,r_hip)
        self.a9.set_data(self.t1,l_knee)
        self.a10.set_data(self.t1,r_knee2)
        self.a11.set_data(self.t1,l_hip2)
        self.a12.set_data(self.t1,r_shoulder2)
        self.a13.set_data(self.t1,l_shoulder2)
        self.a14.set_data(self.t1,r_hip2)
        self.a15.set_data(self.t1,l_knee2)
        self.update_plot_length() # comment out if scrolling plot is not desired
        return self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7, self.a8,  self.a9, self.a10, self.a11, self.a12, self.a13, self.a14, self.a15

    def ldata_callback(self, msg):
        self.lean=msg.x
        self.lean_dot=msg.y       

    def pdata_callback(self, msg):
        self.acc=msg.data
            
    def r_knee_callback(self, msg):
        self.r_knee=msg.data
        
    def l_hip_callback(self, msg):
        self.l_hip=msg.data

    def r_shoulder_callback(self, msg):
        self.r_shoulder=msg.data
               
    def l_shoulder_callback(self, msg):
        self.l_shoulder=msg.data
                                    
    def r_hip_callback(self, msg):
        self.r_hip=msg.data       
                           
    def l_knee_callback(self, msg):
        self.l_knee=msg.data

    def joints_callback(self, msg):
        self.joints=msg.position
        self.r_shoulder2=self.joints[17]
        self.l_shoulder2=self.joints[8]
        self.r_hip2=self.joints[13]
        self.l_hip2=self.joints[4]
        self.r_knee2=self.joints[15]
        self.l_knee2=self.joints[6]
               
    def read_sag_cmds(self):
        # this function reads the commanded positions for the sagittal joints
        p1=self.r_shoulder
        p2=self.l_shoulder
        p3=self.r_hip
        p4=self.l_hip
        p5=self.r_knee
        p6=self.l_knee
        p=[p1,p2,p3,p4,p5,p6]
        return p
        
    def read_imu_data(self):
        a1=self.lean
        a2=self.lean_dot
        a3=self.acc
        a=[a1,a2,a3]
        return a
    
    def read_sag_angs(self):
        t1=self.r_shoulder2
        t2=self.l_shoulder2
        t3=self.r_hip2
        t4=self.l_hip2
        t5=self.r_knee2
        t6=self.l_knee2
        t=[t1,t2,t3,t4,t5,t6]
        return t
    
    def append_new_data(self):
        cmds = self.read_sag_cmds()
        imu = self.read_imu_data()
        angs = self.read_sag_angs()
        
        self.all_cmds = np.vstack([self.all_cmds,cmds])
        self.all_imu = np.vstack([self.all_imu,imu])
        self.all_angs = np.vstack([self.all_angs,angs])
        
    def delete_old_data(self):       
        self.all_cmds = np.delete(self.all_cmds,0,0)
        self.all_imu = np.delete(self.all_imu,0,0)
        self.all_angs = np.delete(self.all_angs,0,0)
            
                                       
if __name__ == "__main__":
    rospy.init_node("plot_data")
    rospy.sleep(1)

    rospy.loginfo("Real-time data plot begins after calibration")
    viz = Visualizer()
    sub2 = rospy.Subscriber('/leandata', Vector3, viz.ldata_callback)
    sub3 = rospy.Subscriber('/pushdata', Float64, viz.pdata_callback)
    sub4 = rospy.Subscriber("/tobe/joint_states", JointState, viz.joints_callback)
    sub5 = rospy.Subscriber('/tobe/l_shoulder_sagittal_joint_position_controller/command', Float64, viz.l_shoulder_callback)
    sub6 = rospy.Subscriber('/tobe/r_shoulder_sagittal_joint_position_controller/command', Float64, viz.r_shoulder_callback)
    sub7 = rospy.Subscriber('/tobe/l_knee_joint_position_controller/command', Float64, viz.l_knee_callback)
    sub8 = rospy.Subscriber('/tobe/r_knee_joint_position_controller/command', Float64, viz.r_knee_callback)
    sub9 = rospy.Subscriber('/tobe/l_hip_sagittal_joint_position_controller/command', Float64, viz.l_hip_callback)
    sub10 = rospy.Subscriber('/tobe/r_hip_sagittal_joint_position_controller/command', Float64, viz.r_hip_callback)
    ani = FuncAnimation(viz.fig, viz.plot_update, init_func=viz.init_plot)
    plt.show(block=True)
    

