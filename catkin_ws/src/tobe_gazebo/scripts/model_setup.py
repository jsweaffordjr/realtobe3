#!/usr/bin/python3

import sys
import random
import std_msgs
import geometry_msgs
import rospy
import math
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *

class SetupRobot:
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_configuration')
        apply_body_wrench = rospy.ServiceProxy('/gazebo/set_model_configuration', ApplyBodyWrench)

    
if __name__ == "__main__":
    rospy.init_node("Setup Robot")
    af = SetupRobot()
    
