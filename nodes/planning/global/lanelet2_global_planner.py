#!/usr/bin/env python3

import logging
import rospy
import numpy as np
from shapely.geometry import LineString, Point

import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from autoware_mini.msg import Path, Waypoint
from geometry_msgs.msg import PoseStamped
from autoware_mini.lanelet2 import load_lanelet2_map


class GlobalPlanner:
    def __init__(self):

        self.goal_point = None

        self.lanelet2_map_path = rospy.get_param('~lanelet2_map_path')

        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_point_callback)
        self.pose_subscriber = rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback)

        self.waypoints_pub = rospy.Publisher('/planning/global_path', Path, queue_size=10, latch=True)

        self.lanelet2_map = load_lanelet2_map(self.lanelet2_map_path)
        
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)

        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        self.output_frame = rospy.get_param('/planning/lanelet2_global_planner/output_frame')
        
        self.speed_limit_kmh = rospy.get_param('/planning/global_planner/speed_limit', 40.0)
        self.max_speed_ms = self.speed_limit_kmh / 3.6

        self.distance_to_goal_limit = rospy.get_param('/planning/lanelet2_global_planner/distance_to_goal_limit')


    def current_pose_callback(self, msg):

        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is not None:
            
            current_pose = Point(self.current_location.x, self.current_location.y)
            goal_pose = Point(self.goal_point.x, self.goal_point.y)

            # TASK 4
            if current_pose.distance(goal_pose) < self.distance_to_goal_limit:
            
                rospy.loginfo(" Goal has been reached !")

                self.publish_waypoints([])
                
                self.path_no_lane_change = []

                self.goal_point = None
                

    def goal_point_callback(self, msg):
        
        if self.current_location is None:
            return
        
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                      msg.pose.orientation.w, msg.header.frame_id)

        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            logging.warning('No route is found !')
            return

        path = route.shortestPath()

        self.path_no_lane_change = path.getRemainingLane(start_lanelet)

        waypoints = self.lanelet_to_waypoints()
        
        if not waypoints:
            self.publish_waypoints([])
            return
        
        waypoints = self.align_goal_point(waypoints)

        self.publish_waypoints(waypoints=waypoints)        


    def align_goal_point(self, waypoints):
        
        goal_point = Point(self.goal_point.x, self.goal_point.y)

        min_distance = float('inf')
        closest_waypoint_index = -1

        for i, wp in enumerate(waypoints):
            
            wp_point = Point(wp.position.x, wp.position.y)
            current_distance = wp_point.distance(goal_point)

            if current_distance < min_distance:
                min_distance = current_distance
                closest_waypoint_index = i
        
        waypoints = waypoints[:closest_waypoint_index + 1]
            

    def lanelet_to_waypoints(self): 
        
        speed = self.max_speed_ms

        waypoints = []

        for lanelet_index, lanelet in enumerate(self.path_no_lane_change):
            
            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref']) / 3.6
                speed = min(speed, self.max_speed_ms)

            center_points = list(lanelet.centerline)
            
            points_to_process = []

            if lanelet_index == 0:    
                points_to_process = center_points
            else:
                points_to_process = center_points[1:]
            
            for point in points_to_process:
                
                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed

                waypoints.append(waypoint)
            
            
        return waypoints            

    def publish_waypoints(self, waypoints): 

        path = Path()        
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints
        self.waypoints_pub.publish(path)

    def run(self):

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = GlobalPlanner()
    node.run()