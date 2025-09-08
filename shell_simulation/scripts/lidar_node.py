#!/usr/bin/env python3

import numpy as np
import sklearn
if sklearn.__version__ == '0.22.2.post1':
    import numpy as np
    if not hasattr(np, 'float'):
        np.float = float
else:
    import numpy as np

from custom_msg.msg import Perception, ObjectInfo, TrafficLightsInfo, Dimensions
import rospy
import threading
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from rosgraph_msgs.msg import Clock
import time
import queue
from sklearn.cluster import DBSCAN
from shell_simulation.kalman_filter import update_tracking
import tf
from std_msgs.msg import Bool
from shell_simulation.lidar_utilities import manual_voxel_filter, manual_ransac, compute_pca_obb, lookup_latest_transform_with_wait

class Lidar:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('Lidar')
        
        
        # Create subscribers
        self.clock_subscriber = rospy.Subscriber(
            '/clock',
            Clock,
            self.clock_callback,
            queue_size=10
        )
        
        self.lidar_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/vlp16_1',
            PointCloud2,
            self.lidar_callback,
            queue_size=10
        )
        
        self.inference_subscriber = rospy.Subscriber(
            'perception/inference',
            Perception,
            self.inference_callback,
            queue_size=10
        )
        
        # Create publishers
        self.marker_publisher = rospy.Publisher('detected_objects', MarkerArray, queue_size=10)
        self.perception_publisher = rospy.Publisher('perception/data', Perception, queue_size=10)
        
        # Initialize TF listener
        self.tf_listener = tf.TransformListener()
        
        # Initialize tracking data
        self.tracking_data = []
        self.detected_objects = []
        self.latest_traffic_lights = []
        
        # Create timer for publishing perception data
        self.publish_timer = rospy.Timer(rospy.Duration(0.2), self.publish_perception_timer_callback)
        
        # Initialize queues and tracking
        self.lidar_queue = queue.Queue(maxsize=10)
        self.time_queue = queue.Queue(maxsize=10)
        self.tracked_objects = {}
        self.next_id = 0
        
        # Start main processing loop in a separate thread
        threading.Thread(target=self.main_loop).start()
        

    def inference_callback(self, msg):
        """Handle incoming perception data from the inference node."""
        
        
        # Store the latest traffic light information
        self.latest_traffic_lights = msg.traffic_lights
        
        

    def publish_perception_timer_callback(self, event):
        """Timer callback to publish perception data periodically."""
        perception_msg = Perception()
        perception_msg.objects = self.detected_objects
        perception_msg.traffic_lights = self.latest_traffic_lights
        
        # Publish the message
        self.perception_publisher.publish(perception_msg)
        

    def clock_callback(self, msg):
        """Handle incoming clock messages."""
        self.currentTime = msg.clock.to_sec()

    def lidar_callback(self, msg):
        if self.lidar_queue.full():
            self.lidar_queue.get_nowait()
        if self.time_queue.full():
            self.time_queue.get_nowait()
        
        # Convert PointCloud2 to numpy array
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2], 1.0])  # Add homogeneous coordinate
        points = np.array(points_list)
        
        self.lidar_queue.put_nowait(points)
        # Store the timestamp from the message header
        self.time_queue.put_nowait(msg.header.stamp)

    def lidar_processing(self):
        if self.lidar_queue.empty() or self.time_queue.empty():
            return
            
        # Get the timestamp for the current LiDAR data
        msg_timestamp = self.time_queue.get_nowait()

        points = self.lidar_queue.get_nowait()

        mask = (points[:, 2] > -1.2) & (points[:, 0] > 0) & (points[:, 2] < 0.5)
        filtered_points = points[mask]
        filtered_points = filtered_points[:, :3]
        if len(filtered_points) == 0:
            return
        downsampled = manual_voxel_filter(filtered_points, 0.2)
        _, obstacles = manual_ransac(downsampled)
        if len(obstacles) == 0:
            return

        labels = DBSCAN(eps=3, min_samples=10).fit(obstacles).labels_
        clusters = {label: [] for label in set(labels) if label != -1}
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(obstacles[i])

        cluster_centers = []
        cluster_obb_data = []
        for cluster in clusters.values():
            cluster_points = np.array(cluster)
            if len(cluster_points) < 10:
                continue
            obb = compute_pca_obb(cluster_points, [4.7, 1.85, 2])
            if obb is None:
                continue
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "ego_vehicle/vlp16_1"
            # Use the timestamp of the LiDAR message
            point_stamped.header.stamp = msg_timestamp
            point_stamped.point.x = obb['center'][0]
            point_stamped.point.y = obb['center'][1]
            point_stamped.point.z = obb['center'][2]

            # Wait for transform to be available at the message timestamp with a small buffer time
            self.tf_listener.waitForTransform("map", "ego_vehicle/vlp16_1", msg_timestamp, rospy.Duration(0.1))
            # Use the transform at the message timestamp
            transformed_point = self.tf_listener.transformPoint("map", point_stamped)

            global_center = np.array([transformed_point.point.x, transformed_point.point.y])
            cluster_centers.append(global_center)
            cluster_obb_data.append((obb, transformed_point))

        marker_array = MarkerArray()
        detected_objects = []
        # Convert ROS Time to seconds for tracking
        current_time_sec = msg_timestamp.to_sec()
        self.tracked_objects = update_tracking(cluster_centers, self.tracked_objects, current_time_sec)
        
        # Create a mapping of centers to their OBB data
        center_to_obb = {tuple(center): obb_data for center, obb_data in zip(cluster_centers, cluster_obb_data)}
        
        # Set a reasonable threshold (e.g. 3.0 meters)
        DIST_THRESHOLD = 3.0

        for obj_id, track_data in self.tracked_objects.items():
            current_pos = track_data['last_global']
            
            # Find closest center within threshold
            closest_center = None
            min_dist = float('inf')
            for center in center_to_obb.keys():
                dist = np.linalg.norm(np.array(center) - current_pos)
                if dist < DIST_THRESHOLD and dist < min_dist:
                    closest_center = center
                    min_dist = dist

            if closest_center is not None:
                obb, transformed_point = center_to_obb.pop(closest_center)
                # Create ObjectInfo message
                obj_info = ObjectInfo()
                obj_info.id = obj_id
                obj_info.classification = 0
                obj_info.pose.position.x = transformed_point.point.x
                obj_info.pose.position.y = transformed_point.point.y
                obj_info.pose.position.z = transformed_point.point.z
                obj_info.pose.orientation.x = obb['quaternion'][0]
                obj_info.pose.orientation.y = obb['quaternion'][1]
                obj_info.pose.orientation.z = obb['quaternion'][2]
                obj_info.pose.orientation.w = obb['quaternion'][3]
                obj_info.dimensions.width = obb['extents'][1]
                obj_info.dimensions.height = obb['extents'][2]
                obj_info.dimensions.depth = obb['extents'][0]
                obj_info.speed = track_data['speed']
                detected_objects.append(obj_info)

                # Create marker
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = msg_timestamp
                marker.ns = "object"
                marker.id = obj_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = transformed_point.point.x
                marker.pose.position.y = transformed_point.point.y
                marker.pose.position.z = transformed_point.point.z
                marker.pose.orientation.x = obb['quaternion'][0]
                marker.pose.orientation.y = obb['quaternion'][1]
                marker.pose.orientation.z = obb['quaternion'][2]
                marker.pose.orientation.w = obb['quaternion'][3]
                marker.scale.x = obb['extents'][0]
                marker.scale.y = obb['extents'][1]
                marker.scale.z = obb['extents'][2]
                marker.color.r = (obj_id * 31 % 255) / 255.0
                marker.color.g = (obj_id * 67 % 255) / 255.0
                marker.color.b = (obj_id * 89 % 255) / 255.0
                marker.color.a = 0.5
                marker.lifetime = rospy.Duration(1)
                marker_array.markers.append(marker)

        self.detected_objects = detected_objects
        self.marker_publisher.publish(marker_array)

    def main_loop(self):
        while not rospy.is_shutdown():
            self.lidar_processing()
            time.sleep(0.01)

def main():
    lidar = Lidar()
    rospy.spin()

if __name__ == '__main__':
    main() 
