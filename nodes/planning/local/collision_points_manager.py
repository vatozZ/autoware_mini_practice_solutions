#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from shapely.geometry import LineString, Polygon

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")

        self.braking_safety_distance_goal = rospy.get_param('~braking_safety_distance_goal')

        self.detected_objects = None

        self.goal_point = None

        self.lock = threading.Lock()

        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/planning/global_path', Path, self.global_path_callback)

    def global_path_callback(self, msg):
        if not msg.waypoints:
            self.goal_point = None
            return

        _waypoints = [(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints]
        self.goal_point = shapely.Point(_waypoints[-1][:2])

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        if self.goal_point is None:
            self.publish_empty_collision(msg=msg)
            return

        if not msg.waypoints:
            self.publish_empty_collision(msg=msg)
            return

        local_path = LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
        local_path_buffer = local_path.buffer(self.safety_box_width/2, cap_style="flat")
        shapely.prepare(local_path_buffer)

        goal_buffer = self.goal_point.buffer(3.0)

        for obj in detected_objects:

            _convex_hull = np.array(obj.convex_hull)
            _convex_hull = _convex_hull.reshape(-1, 3)
            _convex_hull = _convex_hull[:, :2]
            obj_convex_hull = Polygon(_convex_hull)

            if obj_convex_hull.intersects(local_path_buffer):
                _intersection_point = shapely.get_coordinates(obj_convex_hull.intersection(local_path_buffer))

                speed_norm = math.hypot(obj.velocity.x, obj.velocity.y)
                if speed_norm  < self.stopped_speed_limit:
                    category = 3
                else:
                    category = 4

                for x, y in _intersection_point:
                    collision_points = np.append(collision_points, np.array(
                        [(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                          self.braking_safety_distance_obstacle, np.inf, category)],
                        dtype=DTYPE))


        if goal_buffer.intersects(local_path_buffer):
            collision_points = np.append(collision_points,
                                         np.array(
                                             [(self.goal_point.x, self.goal_point.y, 0, 0, 0, 0, self.braking_safety_distance_goal,
                                               np.inf, 1)], dtype=DTYPE
                                         ))

        collision_points_pub = msgify(PointCloud2, collision_points)
        collision_points_pub.header = msg.header
        self.local_path_collision_pub.publish(collision_points_pub)


    def publish_empty_collision(self, msg):
        collision_points = np.array([], dtype=DTYPE)
        empty_collision_points_pub = msgify(PointCloud2, collision_points)
        empty_collision_points_pub.header = msg.header
        self.local_path_collision_pub.publish(empty_collision_points_pub)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()