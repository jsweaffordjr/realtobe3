#!/usr/bin/python3

from threading import Thread
import rospy
import math
import numpy as np
from tobe3_real.teststand1 import Stand
from geometry_msgs.msg import Vector3


if __name__ == "__main__":
    rospy.init_node("Tobe Robot")
    rospy.sleep(1)

    rospy.loginfo("Instantiating Standing Protocol for Tobe")
    stand = Stand()

    while not rospy.is_shutdown():
        rospy.sleep(1)
