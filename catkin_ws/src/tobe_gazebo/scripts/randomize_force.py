#!/usr/bin/python3

import sys
import random
import std_msgs
import geometry_msgs
import rospy
import math
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *

class ApplyForce:
    def __init__(self):
        self.init_time = rospy.get_rostime() # get initial time in seconds, nanoseconds
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._get_state_callback)
        self.initial_push = True
        self.apply_periodic_impulse()

    def randomize_force(self,start_time):
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        random_force = random.uniform(5,40)
        body_name = 'tobe::base_link'
        reference_frame = 'tobe::base_link'
        reference_point = geometry_msgs.msg.Point(x = 0, y = 0, z = 0)
        wrench = geometry_msgs.msg.Wrench(force = geometry_msgs.msg.Vector3( x = random_force, y = 0, z = 0), torque = geometry_msgs.msg.Vector3( x = 0, y = 0, z = 0))
        duration = rospy.Duration(secs = 0, nsecs = 100000000) # 0.1-sec. duration
    
        apply_body_wrench(body_name, reference_frame, reference_point, wrench, start_time, duration)
        rospy.loginfo("Applying %s N...", random_force)
    
    def apply_periodic_impulse(self):
        while not rospy.is_shutdown():
            now = rospy.get_rostime() # get current time
            time = now - self.init_time
            if time.secs > 8 or self.initial_push:
                self.randomize_force(rospy.Time(secs = 0, nsecs = 0))
                self.init_time = now 
                if self.initial_push:
                    self.initial_push = False
            rospy.sleep(1)
    
    def _get_state_callback(self,msg):
        tobe_pose = msg.pose[1] # TOBE robot is second item in environment, so it's second item in pose list
        tobe_height = tobe_pose.position.z
        
        if tobe_height < 0.2: # when torso is below 0.2 m above ground, assume fall has occurred/is occurring
            self.init_time = rospy.get_rostime() # set init_time to current time
    
if __name__ == "__main__":
    rospy.init_node("Force Application")
    af = ApplyForce()
    
