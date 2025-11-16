#!/usr/bin/env python3
from copy import deepcopy

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
import threading
from ros_numpy import numpify
from autoware_mini.msg import Path, Log
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from autoware_mini.geometry import project_vector_to_heading, get_distance_between_two_points_2d


class SpeedPlanner:

    def __init__(self):

        self.default_deceleration = rospy.get_param("default_deceleration")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")
        self.distance_to_car_front = rospy.get_param("distance_to_car_front")

        self.collision_points = None
        self.current_position = None
        self.current_speed = None

        self.lock = threading.Lock()

        self.local_path_pub = rospy.Publisher('local_path', Path, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        collision_points_sub = message_filters.Subscriber('collision_points', PointCloud2, tcp_nodelay=True)
        local_path_sub = message_filters.Subscriber('extracted_local_path', Path, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([collision_points_sub, local_path_sub], queue_size=synchronization_queue_size, slop=synchronization_slop)

        ts.registerCallback(self.collision_points_and_path_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_and_path_callback(self, collision_points_msg, local_path_msg):
        try:
            with self.lock:
                collision_points = numpify(collision_points_msg) if len(collision_points_msg.data) > 0 else np.array([])
                current_position = self.current_position
                current_speed = self.current_speed

            if current_position is None or current_speed is None:
                return

            if collision_points.size == 0:
                self.local_path_pub.publish(local_path_msg)
                return

            local_path = np.array([(waypoint.position.x, waypoint.position.y) for waypoint in local_path_msg.waypoints])
            local_path_lineString = shapely.LineString(local_path)

            raw_distances = []
            distance_of_collision_points = []
            category_list = []
            target_V_list = []
            relative_speed = []

            for i in range(0, collision_points.shape[0]):
                _collision_point = shapely.Point(float(collision_points[i]['x']), float(collision_points[i]['y']))
                category_list.append(int(collision_points[i]['category']))
                _braking_safety_distance = collision_points[i]['distance_to_stop']
                _raw_distance = local_path_lineString.project(_collision_point)

                raw_distances.append(_raw_distance)

                _heading = self.get_heading_at_distance(linestring=local_path_lineString, distance=_raw_distance)

                _velocity_vec = shapely.Point(collision_points[i]['vx'], collision_points[i]['vy'])

                collision_point_velocities = self.project_vector_to_heading(heading_angle=_heading, vector=_velocity_vec)

                reaction_distance = abs(collision_point_velocities) * self.braking_reaction_time
                _stop_dist = max(0.0, _raw_distance - self.distance_to_car_front - _braking_safety_distance - reaction_distance)
                distance_of_collision_points.append(_stop_dist)

                target_V = np.sqrt(max(0, np.power(max(0.0, collision_point_velocities), 2) + 2 * self.default_deceleration * _stop_dist))
                target_V_list.append(target_V)

                relative_speed.append(collision_point_velocities)

            idx = np.argmin(target_V_list)
            target_velocity = target_V_list[idx]
            raw_distance = raw_distances[idx]
            braking_distance = collision_points[idx]['distance_to_stop']

            closest_object_distance = max(0.0, raw_distance - self.distance_to_car_front)
            stopping_point_distance = max(0.0, raw_distance - self.distance_to_car_front - braking_distance)

            collision_point_category = category_list[idx]

            for i, wp in enumerate(local_path_msg.waypoints):
                wp.speed = min(target_velocity, wp.speed)

            path = Path()
            path.header = local_path_msg.header
            path.waypoints = deepcopy(local_path_msg.waypoints)
            path.closest_object_distance = closest_object_distance
            path.closest_object_velocity = relative_speed[idx]
            path.is_blocked = True
            path.stopping_point_distance = stopping_point_distance
            path.collision_point_category = collision_point_category
            self.local_path_pub.publish(path)


        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())


    def get_heading_at_distance(self, linestring, distance):

        point_after_object = linestring.interpolate(distance + 0.1)
        point_before_object = linestring.interpolate(max(0, distance - 0.1))

        return math.atan2(point_after_object.y - point_before_object.y, point_after_object.x - point_before_object.x)


    def project_vector_to_heading(self, heading_angle, vector):

        return vector.x * math.cos(heading_angle) + vector.y * math.sin(heading_angle)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()