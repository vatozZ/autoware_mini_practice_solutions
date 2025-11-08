#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):

        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):

        data = numpify(msg)
        
        points = structured_to_unstructured(data[['x', 'y', 'z', 'label']], dtype=np.float32)
        
        labels = points[:, -1].astype(np.int32)

        frame_id = msg.header.frame_id

        if frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            
        tf_matrix = numpify(transform.transform).astype(np.float32)

        points = points.copy()
        points[:, 3] = 1

        points = points.dot(tf_matrix.T)

        detected_objects = []
        
        for i in np.unique(labels):
            
            if i == -1:
                continue

            mask = (labels == i)
            points3d = points[mask, :3]
            
            if points3d.shape[0] < self.min_cluster_size:
                continue

            centroid = np.mean(points3d, axis=0)
            centroid_x, centroid_y, centroid_z = centroid[0], centroid[1], centroid[2] 
            detected_object = DetectedObject()
            detected_object.centroid.x = centroid_x
            detected_object.centroid.y = centroid_y
            detected_object.centroid.z = centroid_z

            points_2d = MultiPoint(points[mask,:2])
            hull = points_2d.convex_hull
            convex_hull_points = [a for hull in [[x, y, centroid_z] for x, y in hull.exterior.coords] for a in hull]

            detected_object.convex_hull = convex_hull_points
            detected_object.id = i

            detected_object.label = "unknown"
            detected_object.color = BLUE80P
            detected_object.valid = True
            detected_object.position_reliable = True
            detected_object.velocity_reliable = False
            detected_object.acceleration_reliable = False

            detected_objects.append(detected_object)
        
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.output_frame

        detected_object_array = DetectedObjectArray()
        detected_object_array.header = header
        detected_object_array.objects = detected_objects

        self.objects_pub.publish(detected_object_array)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()