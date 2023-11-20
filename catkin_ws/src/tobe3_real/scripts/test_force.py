#!/usr/bin/python3

import sys
import random
import std_msgs
import geometry_msgs
import rospy
import math
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *

class ApplyForce:
    def __init__(self):
        self.init_time = rospy.get_rostime() # get initial time in seconds, nanoseconds
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._get_state_callback)
        self.initial_push = True
        self.pub_force = rospy.Publisher('applied_force', Float64, queue_size=1)
        self.min_force = 30.0
        self.max_force = 50.0
        self.push_count = 0
        self.max_push_count = 11
        self.apply_periodic_impulse()
        
    def randomized_force(self,start_time):
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        random_force = round(random.uniform(self.min_force,self.max_force), 2)
        body_name = 'tobe::base_link'
        reference_frame = 'tobe::base_link'
        reference_point = geometry_msgs.msg.Point(x = 0, y = 0, z = 0)
        wrench = geometry_msgs.msg.Wrench(force = geometry_msgs.msg.Vector3( x = random_force, y = 0.0, z = 0.0), torque = geometry_msgs.msg.Vector3( x = 0.0, y = 0.0, z = 0.0))
        duration = rospy.Duration(secs = 0, nsecs = 100000000) # 0.1-sec. duration
    
        apply_body_wrench(body_name, reference_frame, reference_point, wrench, start_time, duration)
        self.pub_force.publish(random_force)
        rospy.loginfo("Applying %s N...", random_force)
        
    def deterministic_force(self,start_time):
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # this force increases incrementally from min_force to max_force, then goes back to min_force and starts again...
        selected_force = self.min_force + (self.push_count - 1)*((self.max_force-self.min_force)/(self.max_push_count-1))
        
        body_name = 'tobe::base_link'
        reference_frame = 'tobe::base_link'
        reference_point = geometry_msgs.msg.Point(x = 0, y = 0, z = 0)
        wrench = geometry_msgs.msg.Wrench(force = geometry_msgs.msg.Vector3( x = selected_force, y = 0.0, z = 0.0), torque = geometry_msgs.msg.Vector3( x = 0.0, y = 0.0, z = 0.0))
        duration = rospy.Duration(secs = 0, nsecs = 100000000) # 0.1-sec. duration
    
        apply_body_wrench(body_name, reference_frame, reference_point, wrench, start_time, duration)
        self.pub_force.publish(selected_force)
        rospy.loginfo("Applying %s N...", selected_force)
    
    def apply_periodic_impulse(self):
        while not rospy.is_shutdown():
            now = rospy.get_rostime() # get current time
            time = now - self.init_time
            if time.secs > 5 or self.initial_push:
                if self.push_count >= self.max_push_count:
                    self.push_count = 0
                self.push_count += 1
                
                #self.randomize_force(rospy.Time(secs = 0, nsecs = 0))
                self.deterministic_force(rospy.Time(secs = 0, nsecs = 0))
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
    
