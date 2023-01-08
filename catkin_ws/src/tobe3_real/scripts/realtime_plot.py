#!/usr/bin/python3
import roslib
import rospy
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64

# this script subscribes to the /push and /lean topics, generates real-time plot(s)
class Visualizer:
    def __init__(self):
        # pause until after IMU calibration is completed:
        self.calib=False
        rospy.sleep(5)
        self.sub_cal=rospy.Subscriber('/calibration',Vector3,self._calibrate, queue_size=1)
        while self.calib == False:
            rospy.sleep(0.1)
        rospy.sleep(10) 
        
        # plot initialization:
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, sharex=True)
        self.fig.set_size_inches(10, 10)
        self.a1, = self.ax3.plot([], [], label='Forward lean angle', color='black')
        self.a2, = self.ax3.plot([], [], label='Forward lean rate', color='red', linestyle='dashed')
        self.a3, = self.ax4.plot([], [], label='Side lean angle', color='black')
        self.a4, = self.ax4.plot([], [], label='Side lean rate', color='red', linestyle='dashed')
        self.a5, = self.ax2.plot([], [], label='R fro. hip', color='green')
        self.a6, = self.ax1.plot([], [], label='R sag. hip', color='red')
        self.a7, = self.ax1.plot([], [], label='R knee', color='black')
        self.a8, = self.ax1.plot([], [], label='R sag. ankle', color='cyan')
        self.a9, = self.ax2.plot([], [], label='R fro. ankle', color='magenta')
        self.a10, = self.ax2.plot([], [], label='L fro. hip', color='red')
        self.a11, = self.ax1.plot([], [], label='L sag. hip', color='green')
        self.a12, = self.ax1.plot([], [], label='L knee', color='blue')
        self.a13, = self.ax1.plot([], [], label='L sag. ankle', color='magenta')
        self.a14, = self.ax2.plot([], [], label='L fro. ankle', color='cyan')   
        self.a15, = self.ax2.plot([], [], label='R fro. hip cmd', color='green', linestyle='dashed')
        self.a16, = self.ax1.plot([], [], label='R sag. hip cmd', color='red', linestyle='dashed')
        self.a17, = self.ax1.plot([], [], label='R knee cmd', color='black', linestyle='dashed')
        self.a18, = self.ax1.plot([], [], label='R sag. ankle cmd', color='cyan', linestyle='dashed')
        self.a19, = self.ax2.plot([], [], label='R fro. ankle cmd', color='magenta', linestyle='dashed')
        self.a20, = self.ax2.plot([], [], label='L fro. hip cmd', color='red', linestyle='dashed')
        self.a21, = self.ax1.plot([], [], label='L sag. hip cmd', color='green', linestyle='dashed')
        self.a22, = self.ax1.plot([], [], label='L knee cmd', color='blue', linestyle='dashed')
        self.a23, = self.ax1.plot([], [], label='L sag. ankle cmd', color='magenta', linestyle='dashed')
        self.a24, = self.ax2.plot([], [], label='L fro. ankle cmd', color='cyan', linestyle='dashed')      
        self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8, self.t9, self.t10, self.t11 = [], [], [], [], [], [], [], [], [], [], []
        self.t12, self.t13, self.t14, self.t15, self.t16, self.t17, self.t18, self.t19, self.t20, self.t21, self.t22 = [], [], [], [], [], [], [], [], [], [], []
        self.forward_lean, self.forward_lean_dot, self.side_lean, self.side_lean_dot = [], [], [], []
        self.r_f_ankle, self.l_f_ankle, self.r_f_hip, self.l_f_hip, self.r_knee = [], [], [], [], []
        self.r_s_ankle, self.l_s_ankle, self.r_s_hip, self.l_s_hip, self.l_knee = [], [], [], [], []
        self.r_f_ankle_cmd, self.l_f_ankle_cmd, self.r_f_hip_cmd, self.l_f_hip_cmd, self.r_knee_cmd = [], [], [], [], []
        self.r_s_ankle_cmd, self.l_s_ankle_cmd, self.r_s_hip_cmd, self.l_s_hip_cmd, self.l_knee_cmd = [], [], [], [], []
        self.initial_time = rospy.get_time()
        self.plot_length = 10 # duration of the plot window
        self.t_start=0
        self.t_end=self.plot_length
        self.fig.suptitle('Real-time plots')
        
    def _calibrate(self,msg):
        x=msg.x
        y=msg.y
        z=msg.z
        if x > 2 and y > 2 and z > 2:
            self.calib=True
            
    def init_plot(self):
        self.ax1.set_title('Sagittal joint angles')
        self.ax1.set_xlim(0,self.plot_length)
        self.ax1.set_ylim(-1.5,1.5)
        self.ax1.legend(loc='center left')
        self.ax2.set_title('Frontal joint angles')
        self.ax2.set_xlim(0,self.plot_length)
        self.ax2.set_ylim(-1,1)
        self.ax2.legend(loc='center left')
        self.ax3.set_title('Torso angle [backward (+)] and rate')
        self.ax3.set_xlim(0,self.plot_length)
        self.ax3.set_ylim(-0.5,0.5)
        self.ax3.legend(loc='center left')
        self.ax4.set_title('Torso angle [right (+)] and rate')
        self.ax4.set_xlim(0,self.plot_length)
        self.ax4.set_ylim(-0.5,0.5)
        self.ax4.legend(loc='center left')
        return self.ax1, self.ax2 #, self.ax3        

    def update_plot_length(self):
        self.ax1.set_xlim(self.t_start,self.t_end)
        self.ax2.set_xlim(self.t_start,self.t_end)
        self.ax3.set_xlim(self.t_start,self.t_end)
        self.ax4.set_xlim(self.t_start,self.t_end)
        return self.ax1, self.ax2, self.ax3  
                
    def plot_update(self, frame):
        self.a1.set_data(self.t21,self.forward_lean)
        self.a2.set_data(self.t21,self.forward_lean_dot)
        self.a3.set_data(self.t22,self.side_lean)
        self.a4.set_data(self.t22,self.side_lean_dot)
        self.a5.set_data(self.t1,self.r_f_hip)
        self.a6.set_data(self.t2,self.r_s_hip)
        self.a7.set_data(self.t3,self.r_knee)
        self.a8.set_data(self.t4,self.r_s_ankle)
        self.a9.set_data(self.t5,self.r_f_ankle)
        self.a10.set_data(self.t6,self.l_f_hip)
        self.a11.set_data(self.t7,self.l_s_hip)
        self.a12.set_data(self.t8,self.l_knee)
        self.a13.set_data(self.t9,self.l_s_ankle)
        self.a14.set_data(self.t10,self.r_f_ankle)
        self.a15.set_data(self.t11,self.r_f_hip_cmd)
        self.a16.set_data(self.t12,self.r_s_hip_cmd)
        self.a17.set_data(self.t13,self.r_knee_cmd)
        self.a18.set_data(self.t14,self.r_s_ankle_cmd)
        self.a19.set_data(self.t15,self.r_f_ankle_cmd)
        self.a20.set_data(self.t16,self.l_f_hip_cmd)
        self.a21.set_data(self.t17,self.l_s_hip_cmd)
        self.a22.set_data(self.t18,self.l_knee_cmd)
        self.a23.set_data(self.t19,self.l_s_ankle_cmd)
        self.a24.set_data(self.t20,self.r_f_ankle_cmd)
        self.update_plot_length() # comment out if scrolling plot is not desired
        return self.a5, self.a6, self.a7, self.a8, self.a9, self.a10, self.a11, self.a12, self.a13, self.a14, self.a15, self.a16, self.a17, self.a18, self.a19, self.a20, self.a21, self.a22, self.a23, self.a24, self.a1, self.a2, self.a3, self.a4

        
    def ldata_callback1(self, msg):
        tnow=rospy.get_time()-self.initial_time      
        self.t21.append(tnow)
        self.forward_lean.append(msg.x)
        self.forward_lean_dot.append(msg.y)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t21.pop(0)
            self.forward_lean.pop(0)
            self.forward_lean_dot.pop(0)
            self.t_end=tnow
            self.t_start=tnow-self.plot_length

    def ldata_callback2(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t22.append(tnow)
        self.side_lean.append(msg.x)
        self.side_lean_dot.append(msg.y)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t22.pop(0)
            self.side_lean.pop(0)
            self.side_lean_dot.pop(0)

    def r_f_hip_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t1.append(tnow)
        self.r_f_hip.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t1.pop(0)
            self.r_f_hip.pop(0)
        
    def r_s_hip_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t2.append(tnow)
        self.r_s_hip.append(msg.data)  

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t2.pop(0)
            self.r_s_hip.pop(0)
            
    def r_knee_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t3.append(tnow)
        self.r_knee.append(msg.data) 

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t3.pop(0)
            self.r_knee.pop(0)
               
    def r_s_ankle_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t4.append(tnow)
        self.r_s_ankle.append(msg.data)

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t4.pop(0)
            self.r_s_ankle.pop(0)

    def r_f_ankle_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t5.append(tnow)
        self.r_f_ankle.append(msg.data)        
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t5.pop(0)
            self.r_f_ankle.pop(0)            
    
    def l_f_hip_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t6.append(tnow)
        self.l_f_hip.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t6.pop(0)
            self.l_f_hip.pop(0)

    def l_s_hip_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t7.append(tnow)
        self.l_s_hip.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t7.pop(0)
            self.l_s_hip.pop(0)
        
    def l_knee_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t8.append(tnow)
        self.l_knee.append(msg.data)  

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t8.pop(0)
            self.l_knee.pop(0)
    
    def l_s_ankle_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t9.append(tnow)
        self.l_s_ankle.append(msg.data) 

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t9.pop(0)
            self.l_s_ankle.pop(0)
               
    def l_f_ankle_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t10.append(tnow)
        self.l_f_ankle.append(msg.data)

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t10.pop(0)
            self.l_f_ankle.pop(0)                                   
    
    def r_f_hip_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t11.append(tnow)
        self.r_f_hip_cmd.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t11.pop(0)
            self.r_f_hip_cmd.pop(0)
        
    def r_s_hip_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t12.append(tnow)
        self.r_s_hip_cmd.append(msg.data)  

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t12.pop(0)
            self.r_s_hip_cmd.pop(0)
            
    def r_knee_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t13.append(tnow)
        self.r_knee_cmd.append(msg.data) 

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t13.pop(0)
            self.r_knee_cmd.pop(0)
               
    def r_s_ankle_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t14.append(tnow)
        self.r_s_ankle_cmd.append(msg.data)

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t14.pop(0)
            self.r_s_ankle_cmd.pop(0)

    def r_f_ankle_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t15.append(tnow)
        self.r_f_ankle_cmd.append(msg.data)        
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t15.pop(0)
            self.r_f_ankle_cmd.pop(0)            
    
    def l_f_hip_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t16.append(tnow)
        self.l_f_hip_cmd.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t16.pop(0)
            self.l_f_hip_cmd.pop(0)

    def l_s_hip_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t17.append(tnow)
        self.l_s_hip_cmd.append(msg.data)
        
        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length: 
            self.t17.pop(0)
            self.l_s_hip_cmd.pop(0)
        
    def l_knee_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t18.append(tnow)
        self.l_knee_cmd.append(msg.data)  

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t18.pop(0)
            self.l_knee_cmd.pop(0)
    
    def l_s_ankle_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t19.append(tnow)
        self.l_s_ankle_cmd.append(msg.data) 

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t19.pop(0)
            self.l_s_ankle_cmd.pop(0)
               
    def l_f_ankle_cmd_callback(self, msg):
        tnow=rospy.get_time()-self.initial_time
        self.t20.append(tnow)
        self.l_f_ankle_cmd.append(msg.data)

        # for scrolling plot: (NOTE: if scrolling plot is not desired, comment out the remainder of this callback)
        # after the length of plot window has passed, remove the oldest data point each time a new one comes in
        if tnow > self.plot_length:
            self.t20.pop(0)
            self.l_f_ankle_cmd.pop(0) 
                                       
if __name__ == "__main__":
    rospy.init_node("plot_data")
    rospy.sleep(1)

    rospy.loginfo("Real-time data plot begins after calibration")
    viz = Visualizer()
    sub2 = rospy.Subscriber('/leandata1', Vector3, viz.ldata_callback1)
    sub3 = rospy.Subscriber('/leandata2', Vector3, viz.ldata_callback2)
    sub4 = rospy.Subscriber('/realtobe/r_hip_frontal/angle', Float64, viz.r_f_hip_callback)
    sub5 = rospy.Subscriber('/realtobe/r_hip_sagittal/angle', Float64, viz.r_s_hip_callback) 
    sub6 = rospy.Subscriber('/realtobe/r_knee/angle', Float64, viz.r_knee_callback) 
    sub7 = rospy.Subscriber('/realtobe/r_ankle_sagittal/angle', Float64, viz.r_s_ankle_callback) 
    sub8 = rospy.Subscriber('/realtobe/r_ankle_frontal/angle', Float64, viz.r_f_ankle_callback)
    sub9 = rospy.Subscriber('/realtobe/l_hip_frontal/angle', Float64, viz.l_f_hip_callback)
    sub10 = rospy.Subscriber('/realtobe/l_hip_sagittal/angle', Float64, viz.l_s_hip_callback)
    sub11 = rospy.Subscriber('/realtobe/l_knee/angle', Float64, viz.l_knee_callback) 
    sub12 = rospy.Subscriber('/realtobe/l_ankle_sagittal/angle', Float64, viz.l_s_ankle_callback) 
    sub13 = rospy.Subscriber('/realtobe/l_ankle_frontal/angle', Float64, viz.l_f_ankle_callback) 
    sub14 = rospy.Subscriber('/realtobe/r_hip_frontal/command', Float64, viz.r_f_hip_cmd_callback)
    sub15 = rospy.Subscriber('/realtobe/r_hip_sagittal/command', Float64, viz.r_s_hip_cmd_callback) 
    sub16 = rospy.Subscriber('/realtobe/r_knee/command', Float64, viz.r_knee_cmd_callback) 
    sub17 = rospy.Subscriber('/realtobe/r_ankle_sagittal/command', Float64, viz.r_s_ankle_cmd_callback) 
    sub18 = rospy.Subscriber('/realtobe/r_ankle_frontal/command', Float64, viz.r_f_ankle_cmd_callback)
    sub19 = rospy.Subscriber('/realtobe/l_hip_frontal/command', Float64, viz.l_f_hip_cmd_callback)
    sub20 = rospy.Subscriber('/realtobe/l_hip_sagittal/command', Float64, viz.l_s_hip_cmd_callback)
    sub21 = rospy.Subscriber('/realtobe/l_knee/command', Float64, viz.l_knee_cmd_callback) 
    sub22 = rospy.Subscriber('/realtobe/l_ankle_sagittal/command', Float64, viz.l_s_ankle_cmd_callback) 
    sub23 = rospy.Subscriber('/realtobe/l_ankle_frontal/command', Float64, viz.l_f_ankle_cmd_callback) 
    
    ani = FuncAnimation(viz.fig, viz.plot_update, init_func=viz.init_plot)
    plt.show(block=True)
    

