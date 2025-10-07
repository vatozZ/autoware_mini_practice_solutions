#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')

message = rospy.get_param('~message', 'Hello World!')

ros_rate = rospy.get_param('~rate', 1)

rate = rospy.Rate(ros_rate)

pub = rospy.Publisher('/message', String, queue_size=10)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()

