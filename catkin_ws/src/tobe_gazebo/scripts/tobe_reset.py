#!/usr/bin/python3
import rospy
import roslaunch
import roslib
import math
from gazebo_msgs.msg import ModelStates
import geometry_msgs

class FallReset:
    def __init__(self):
        launch_folder_path = "/home/jerry/realtobe3/catkin_ws/src/tobe_gazebo/launch/" # change this for your setup!
        rospy.sleep(3)
        self.fallen = False
        self.tobe_height = 0.35
        self.file_path0=launch_folder_path+"tobe_setup.launch"
        self.file_path1=launch_folder_path+"tobe_reset_world.launch"
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [self.file_path0])
        self.launch2 = self.launch
        self.run_number = 1
        self.launch.start()
        rospy.loginfo("Tobe_reset node running")
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._get_state_callback)
        
        self.reset_sim()
    
    def _get_state_callback(self,msg):
        tobe_pose = msg.pose[1] # TOBE robot is second item in environment, so it's second item in pose list
        self.tobe_height = tobe_pose.position.z
        if self.tobe_height < 0.2: # when torso height is below 0.2 m, assume fall
            self.fallen = True

    def reset_sim(self):
        while not rospy.is_shutdown():
            if self.fallen: 
                rospy.loginfo("Fall detected. Resetting environment...")
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
            else:
                rospy.sleep(1)

if __name__ == "__main__":
    rospy.init_node("fall_reset")
    rospy.sleep(1)
    fr = FallReset()
