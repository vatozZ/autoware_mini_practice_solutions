#!/usr/bin/env python3


import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
from sklearn.cluster import DBSCAN

class PointsClusterer:
    def __init__(self):


        self.pub_points_clustered = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True, )

        self.sub_points_filtered = rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)


        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')

        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size) 
        


    def points_callback(self, msg):
        
        data = numpify(msg)

        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)

        labels = self.clusterer.fit_predict(points)
        
        assert points.shape[0] == labels.shape[0], 'Number of points and labels do not match !'

        label_filter_array = labels != -1

        filtered_points = points[label_filter_array]
        filtered_labels = labels[label_filter_array]


        filtered_labels = filtered_labels.reshape(-1, 1)
        filtered_points = filtered_points.reshape(-1, 3)

        points_labeled = np.concatenate((filtered_points, filtered_labels), axis=1)

        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
            ]))

        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header = msg.header
        cluster_msg.header.stamp = msg.header.stamp
        
        self.pub_points_clustered.publish(cluster_msg)






    def run(self):

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()