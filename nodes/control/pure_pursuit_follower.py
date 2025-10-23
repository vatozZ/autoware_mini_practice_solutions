#!/usr/bin/env python3

import rospy
import shapely
from autoware_mini.msg import Path, VehicleCmd
from geometry_msgs.msg import PoseStamped
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
import numpy as np

from scipy.interpolate import interp1d


class PurePursuitFollower:
    def __init__(self):

        # Parameters

        self.lookahead_distance = rospy.get_param('/control/pure_pursuit_follower/lookahead_distance')
        self.wheel_base = rospy.get_param('/vehicle/wheel_base')

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)

        self.current_pose = None
        self.path_linestring = None
        self.steering_angle = 0.0
        self.velocity = 0.0
        self.distance_to_velocity_interpolator = None
        
        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        
        self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        
        #prepare path - creates spatial tree, making the spatial queries more efficient
        
        prepare(self.path_linestring)

        # Waypoint Operations

        # Create a distance-to-velocity interpolator for the path
        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0) ** 2, axis=1)))
        # add 0 distance in the beginning
        distances = np.insert(distances, 0, 0)
        # Extract velocity values at waypoints
        velocities = np.array([w.speed for w in msg.waypoints])

        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear')


    def current_pose_callback(self, msg):
        # project -> en yakın mesafeyi döndürüyor.

        self.current_pose = msg

        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        
        if self.path_linestring == None:
            return
        
        d_ego_from_path_start = self.path_linestring.project(current_pose)

        L = self.wheel_base

        ld = self.lookahead_distance

        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])


        look_ahaed_point = self.path_linestring.interpolate(d_ego_from_path_start + ld)

        # lookahead point heading calculation
        lookahead_heading = np.arctan2(look_ahaed_point.y - current_pose.y, look_ahaed_point.x - current_pose.x)

        ld = shapely.distance(look_ahaed_point, Point(current_pose.x, current_pose.y))

        # project -> noktanın lineString üzerindeki en yakın noktasının mesafesini bulur, aracın yolun başlangıcından itibaren
        # olan kat ettiği mesafeyi bulur.

        # interpolate -> bir mesafe verdiğimizde, linestring üzerinde başlangıçtan itibaren ilerleyip o noktayı bulur.

        alpha = lookahead_heading - heading

        steering_angle = np.arctan(2 * L * np.sin(alpha) / ld)

        self.steering_angle = steering_angle
        self.velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)


    def run(self):

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if not self.current_pose is None and not self.distance_to_velocity_interpolator is None:

                vehicle_cmd = VehicleCmd()
                vehicle_cmd.ctrl_cmd.steering_angle = self.steering_angle
                vehicle_cmd.ctrl_cmd.linear_velocity = self.velocity

                vehicle_cmd.header.stamp = self.current_pose.header.stamp
                vehicle_cmd.header.frame_id = 'base_link'

                self.vehicle_cmd_pub.publish(vehicle_cmd)

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()