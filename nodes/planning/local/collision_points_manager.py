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
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from autoware_mini.msg import TrafficLightResult, TrafficLightResultArray

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
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")

        self.braking_safety_distance_goal = rospy.get_param('~braking_safety_distance_goal')

        self.detected_objects = None

        self.goal_point = None

        self.lock = threading.Lock()

        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)
        
        self.all_stoplines = self.get_stoplines(lanelet2_map)
        self.trafficlights = self.get_stoplines_trafficlights(lanelet2_map)

        self.stopline_statuses = {}
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/planning/global_path', Path, self.global_path_callback)

    @staticmethod
    def get_stoplines(lanelet2_map):
        stoplines = {}
        for line in lanelet2_map.lineStringLayer:
            if line.attributes:
                if line.attributes["type"] == "stop_line":
                    stoplines[line.id] = LineString([(p.x, p.y) for p in line])

        return stoplines

    @staticmethod
    def get_stoplines_trafficlights(lanelet2_map):
        signals = {}

        for reg_el in lanelet2_map.regulatoryElementLayer:
            if reg_el.attributes["subtype"] == "traffic_light":
                linkId = reg_el.parameters["ref_line"][0].id

                for tfl in reg_el.parameters["refers"]:
                    tfl_height = float(tfl.attributes["height"])

                    plId = tfl.id

                    traffic_light_data = {'top_left': [tfl[0].x, tfl[0].y, tfl[0].z + tfl_height],
                                        'top_right': [tfl[1].x, tfl[1].y, tfl[1].z + tfl_height],
                                        'bottom_left': [tfl[0].x, tfl[0].y, tfl[0].z],
                                        'bottom_right': [tfl[1].x, tfl[1].y, tfl[1].z]}

                    signals.setdefault(linkId, {}).setdefault(plId, traffic_light_data)

        return signals

    def traffic_light_status_callback(self, msg):
        for result in msg.results:
            self.stopline_statuses[result.stopline_id] = result.recognition_result


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

        for stopline_id, stopline_geom in self.all_stoplines.items():

            if stopline_id not in self.stopline_statuses:
                continue
            
            if self.stopline_statuses[stopline_id] == 1:
                continue
            
            if stopline_geom.intersects(local_path_buffer):
                
                _intersection_point = shapely.get_coordinates(stopline_geom.intersection(local_path_buffer))

                for x, y in _intersection_point:

                    collision_points = np.append(collision_points, np.array(
                        [(x, y, 0, 0, 0, 0, self.braking_safety_distance_stopline, np.inf, 2)], dtype=DTYPE))

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