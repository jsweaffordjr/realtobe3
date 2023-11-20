#!/usr/bin/python3
import os
import time
import datetime
import rospy
import roslaunch
import roslib
import rosbag
import math
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64
import geometry_msgs

class FallReset:
    def __init__(self):
        launch_folder_path = "/home/jerry/realtobe3/catkin_ws/src/tobe3_real/launch/" # change this for your setup!
        rospy.sleep(3)
        self.fallen = False
        self.tobe_height = 0.35
        self.file_path0=launch_folder_path+"simtobe_setup.launch"
        self.file_path1=launch_folder_path+"simtobe_reset_world.launch"
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [self.file_path0])
        self.launch2 = self.launch
        self.run_number = 1
        self.launch.start()
        rospy.loginfo("Tobe_reset node running")
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._get_state_callback)
        self.sub2 = rospy.Subscriber('/applied_force', Float64, self._update_datafile)
        
        run_date = datetime.datetime.now()
        run_time = str(run_date.month) + "-" + str(run_date.day) + "-" + str(run_date.hour) + str(run_date.minute)
        results_dir = "/home/jerry/gazebo_tests/"
        datafile_name = f"{run_time}__0109_run2" # enter model folder name here
        datapath = os.path.join(results_dir, datafile_name)
        suffix = ".txt"
        self.filename = datapath+suffix # path to folder in which to save data
        
        self.latest_force = 0.0
        self.init_time = datetime.datetime.now() # get initial time in seconds, nanoseconds
        
        self.reset_sim()
    
    def _update_datafile(self,msg):
        self.latest_force = msg.data
        current_time = round((datetime.datetime.now() - self.init_time).total_seconds(), 1)
        print(f"Applying {self.latest_force}-N force at t = {current_time} sec.\n", file=open(self.filename, 'a'))
        
    
    def _get_state_callback(self,msg):
        tobe_pose = msg.pose[1] # TOBE robot is second item in environment, so it's the second item in pose list
        self.tobe_height = tobe_pose.position.z
        if self.tobe_height < 0.2: # when torso height is below 0.2 m, assume fall
            self.fallen = True

    def reset_sim(self):
        while not rospy.is_shutdown():
            if self.fallen: 
                current_time = round((datetime.datetime.now() - self.init_time).total_seconds(), 1)
                rospy.loginfo("Fall detected. Resetting environment...")
                print(f"Fall detected at t = {current_time}. Resetting environment...\n", file=open(self.filename, 'a'))
                self.launch.shutdown() 
                if self.run_number > 1:
                    self.launch2.shutdown()
                self.run_number += 1
                rospy.sleep(3)
                uuid1 = roslaunch.rlutil.get_or_generate_uuid(None, False)
                uuid2 = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid1)
                roslaunch.configure_logging(uuid2)
                self.launch = roslaunch.parent.ROSLaunchParent(uuid1, [self.file_path1])
                self.launch2 = roslaunch.parent.ROSLaunchParent(uuid2, [self.file_path0])
                self.launch.start()
                rospy.sleep(2)
                self.launch2.start()     
                self.fallen = False
                rospy.sleep(3)
                self.init_time = datetime.datetime.now()
            else:
                rospy.sleep(1)

if __name__ == "__main__":
    rospy.init_node("fall_reset")
    rospy.sleep(1)
    fr = FallReset()
