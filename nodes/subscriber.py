import rospy
from std_msgs.msg import String


def callback(msg):
    print(msg.data)


rospy.init_node('subscriber')

sub = rospy.Subscriber('/message', String, callback)

rospy.spin()