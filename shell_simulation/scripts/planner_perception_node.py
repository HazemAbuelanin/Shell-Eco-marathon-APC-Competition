#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, Point, Quaternion
import math
import numpy as np
import os
import rospkg
import pandas as pd
from custom_msg.msg import Perception, ObjectInfo, TrafficLightsInfo, Dimensions
from scipy.interpolate import CubicSpline

class PlannerPerceptionNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('planner_perception')

        # Global parameters
        self.VEHICLE_DETECTION_RANGE = 50.0
        self.TRAFFIC_LIGHT_DETECTION_RANGE = 20.0
        self.INTERSECTION_DETECTION_RANGE = 30.0  # 30 meters range for intersection detection

        # Initialize lists and data
        self.current_pose = None
        self.current_speed = 0.0
        self.vehicles_list = []
        self.waypoints = []  # Initialize empty waypoints list
        self.vehicle_in_right_lane = None
        self.vehicle_in_same_lane = None
        self.vehicle_in_left_lane = None
        self.vehicle_in_same_lane_ahead = None
        self.vehicle_in_same_lane_behind = None
        self.traffic_light_list = self.read_traffic_lights()
        self.traffic_light_state = 4  # Default state (unknown)
        self.intersection_centroids = self.read_intersection_centroids()
        self.nearest_intersection = None
        self.is_safe = True  # Flag to indicate if it's safe in intersection

        # Create subscribers
        rospy.Subscriber(
            '/carla/ego_vehicle/odometry',
            Odometry,
            self.odometry_callback,
            queue_size=10
        )

        rospy.Subscriber(
            '/planner/trajectory/global',
            Path,
            self.trajectory_callback,
            queue_size=10
        )

        rospy.Subscriber(
            '/perception/data',
            Perception,
            self.objects_callback,
            queue_size=10
        )

        # Create publishers
        self.perception_pub = rospy.Publisher(
            '/planner/perception',
            Perception,
            queue_size=10
        )
        self.intersection_flag_pub = rospy.Publisher(
            '/planner/flag/intersection',
            Bool,
            queue_size=10
        )
        self.safe_in_intersection_flag_pub = rospy.Publisher(
            '/planner/flag/safe_in_intersection',
            Bool,
            queue_size=10
        )

        # Internal state for flags
        self._intersection_flag = False
        self._safe_in_intersection_flag = True

        # Buffer logic state for safe_in_intersection_flag
        self._safe_in_intersection_buffer = 0
        self._safe_in_intersection_last_published = True

        # Timers for each publisher
        rospy.Timer(rospy.Duration(0.1), self.publish_perception_data)  # 10 Hz
        rospy.Timer(rospy.Duration(0.1), self.publish_intersection_flag)  # 10 Hz
        rospy.Timer(rospy.Duration(0.1), self.publish_safe_in_intersection_flag)  # 10 Hz
        

    def odometry_callback(self, msg):
        """Callback for vehicle odometry data"""
        self.current_pose = msg.pose
        self.current_speed = math.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2 +
            msg.twist.twist.linear.z**2
        ) * 3.6  # Convert to km/h

    def trajectory_callback(self, msg):
        """Callback for trajectory path data"""
        self.waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.waypoints.append((x, y))

    def objects_callback(self, msg):
        """Callback for perception data."""
        self.vehicles_list.clear()

        # Process objects in the Perception message
        for obj in msg.objects:
            data = {
                'id': obj.id,
                'location': {
                    'x': obj.pose.position.x,
                    'y': obj.pose.position.y,
                    'z': obj.pose.position.z
                },
                'speed': obj.speed
            }
            self.vehicles_list.append(data)
        
        # Process traffic lights in the Perception message if traffic lights exist
        for tl in msg.traffic_lights:
            self.traffic_light_state = tl.state
        

    def read_traffic_lights(self):
        """Read traffic lights data using rospkg and pandas"""
        traffic_lights = []
        try:
            # Get package path using rospkg
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('shell_simulation')
            traffic_lights_file = os.path.join(package_path, 'data', 'traffic_lights_info.csv')

            # Read CSV using pandas
            df = pd.read_csv(traffic_lights_file)
            for _, row in df.iterrows():
                traffic_lights.append({
                    'id': int(row[0]),
                    'position': (float(row[1]), float(row[2]), float(row[3])),
                })
        except Exception as e:
            rospy.logerr(f"Error reading traffic lights: {str(e)}")
            
        return traffic_lights

    def read_intersection_centroids(self):
        """Read intersection centroids from CSV file"""
        centroids = []
        try:
            # Get package path using rospkg
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('shell_simulation')
            centroids_file = os.path.join(package_path, 'data', 'centroids.csv')

            # Read CSV using pandas
            df = pd.read_csv(centroids_file)
            for _, row in df.iterrows():
                x = float(row['x'])
                y = float(row['y'])
                y = -y
                centroids.append({
                    'x': x,
                    'y': y
                })
        except Exception as e:
            rospy.logerr(f"Error reading intersection centroids: {str(e)}")
            
        return centroids

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def is_in_zone(self, center_x, center_y, obj_x, obj_y):
        side = 6  # 4 meters side length for the intersection zone
        min_x = center_x - side
        max_x = center_x + side
        min_y = center_y - side
        max_y = center_y + side

        # If ego position is available, check distance to ego
        if self.current_pose is not None:
            ego_x = self.current_pose.pose.position.x
            ego_y = self.current_pose.pose.position.y
            distance_to_ego = self.get_distance(obj_x, obj_y, ego_x, ego_y)
            if distance_to_ego < 2:
                return False  # Object is too close to ego, consider it out of zone

        return min_x <= obj_x <= max_x and min_y <= obj_y <= max_y

    def check_intersection_zone_vehicles(self):
        """Check if any vehicle is in the intersection zone near the ego vehicle."""
        if self.current_pose is None or not self.nearest_intersection:
            return

        ego_location = self.current_pose.pose.position
        zone_center_x = self.nearest_intersection['x']
        zone_center_y = self.nearest_intersection['y']

        for v in self.vehicles_list:
            obj_x = v['location']['x']
            obj_y = v['location']['y']
            if self.is_in_zone(zone_center_x, zone_center_y, obj_x, obj_y):
                return False  # Vehicle is in the intersection zone
        return True  # No vehicles in the intersection zone

    def get_frenet_coord(self, x, y, total_wps):
        """Convert Cartesian coordinates to Frenet coordinates using cubic spline interpolation."""
        if len(total_wps) < 2:
            return 0.0, 0.0

        # Extract x and y coordinates from waypoints
        wp_x = [wp[0] for wp in total_wps]
        wp_y = [wp[1] for wp in total_wps]

        # Create cubic splines for x and y
        spline_x = CubicSpline(range(len(wp_x)), wp_x)
        spline_y = CubicSpline(range(len(wp_y)), wp_y)

        # Find the closest point on the spline
        min_dist = float('inf')
        closest_s = 0.0
        for s in np.linspace(0, len(wp_x) - 1, num=100):  # Sample along the spline
            proj_x = spline_x(s)
            proj_y = spline_y(s)
            dist = self.get_distance(x, y, proj_x, proj_y)
            if dist < min_dist:
                min_dist = dist
                closest_s = s

        # Calculate frenet_d (lateral distance)
        proj_x = spline_x(closest_s)
        proj_y = spline_y(closest_s)
        frenet_d = self.get_distance(x, y, proj_x, proj_y)

        # Determine the sign of frenet_d
        dx = spline_x(closest_s + 1e-3) - spline_x(closest_s)
        dy = spline_y(closest_s + 1e-3) - spline_y(closest_s)
        cross = np.cross([dx, dy, 0], [x - proj_x, y - proj_y, 0])
        if cross[2] < 0:
            frenet_d *= -1

        # Calculate frenet_s (longitudinal distance)
        frenet_s = closest_s

        return frenet_s, frenet_d

    def detect_vehicles(self,  vehicle_width=1, vehicle_length=1):
        # Detect vehicles in the same lane and adjacent lanes using vehicle dimensions.
        same_lane_width = 1
        lane_width = 4

        self.vehicle_in_left_lane = None
        self.vehicle_in_same_lane = None

        if self.current_pose is None:
            return None

        ego_location = self.current_pose.pose.position
        vehicles = self.vehicles_list

        ego_s, ego_d = self.get_frenet_coord(
            ego_location.x, ego_location.y, self.waypoints)

        for v in vehicles:
            obstacle_location = v['location']
            obs_s, obs_d = self.get_frenet_coord(
                obstacle_location['x'], obstacle_location['y'], self.waypoints)
            s_diff = obs_s - ego_s
            d_diff = obs_d - ego_d

            # Adjust detection to account for the full dimensions of the vehicle
            if s_diff > -vehicle_length / 2 and s_diff < self.VEHICLE_DETECTION_RANGE + vehicle_length / 2:
                if -same_lane_width / 2 - vehicle_width / 2 <= d_diff <= same_lane_width / 2 + vehicle_width / 2:
                    self.vehicle_in_same_lane = v
            if d_diff > lane_width / 2 and d_diff < lane_width:
                self.vehicle_in_left_lane = v  
        
    def safe_to_maneuver(self):
        if self.vehicle_in_left_lane:
            return False
        else:
            return True
        
    def detect_traffic(self):
        if self.current_pose is None:
            return None

        ego_location = self.current_pose.pose.position
        closest_traffic_light = None
        min_distance = float('inf')

        for traffic_light in self.traffic_light_list:
            traffic_light_pos = traffic_light['position']
            distance = self.get_distance(
                ego_location.x, ego_location.y,
                traffic_light_pos[0], traffic_light_pos[1]
            )

            if distance < self.TRAFFIC_LIGHT_DETECTION_RANGE and distance < min_distance:
                min_distance = distance
                closest_traffic_light = traffic_light

        return closest_traffic_light

    def check_intersection_proximity(self):
        """
        Return (is_near, nearest_intersection).
        is_near: True if vehicle is near any intersection, else False.
        nearest_intersection: dict with 'x' and 'y' of the closest intersection, or None.
        """
        if self.current_pose is None or not self.intersection_centroids:
            return False, None

        ego_location = self.current_pose.pose.position  
        min_distance = float('inf')
        nearest_intersection = None

        for intersection in self.intersection_centroids:
            distance = self.get_distance(
                ego_location.x, ego_location.y,
                intersection['x'], intersection['y']
            )
            if distance < min_distance:
                min_distance = distance
                nearest_intersection = intersection

        is_near = min_distance < self.INTERSECTION_DETECTION_RANGE
        if not is_near:
            nearest_intersection = None
        return is_near, nearest_intersection

    def update_flags(self):
        is_near, nearest = self.check_intersection_proximity()
        self.nearest_intersection = nearest
        if is_near:
            self._intersection_flag = True
            self._safe_in_intersection_flag = self.check_intersection_zone_vehicles()
        else:
            self._intersection_flag = False
            self._safe_in_intersection_flag = True

    def publish_perception_data(self, event):
        if not self.current_pose:
            return

        self.update_flags()  # Always update flags before publishing

        # Create Perception message
        msg = Perception()
        msg.objects = []
        msg.traffic_lights = []

        # Add closest vehicle if detected
        self.detect_vehicles()
        vehicle = self.vehicle_in_same_lane
        if vehicle:
            obj_info = ObjectInfo()
            obj_info.id = vehicle['id']
            obj_info.pose = Pose(
                position=Point(
                    x=vehicle['location']['x'],
                    y=vehicle['location']['y'],
                    z=vehicle['location']['z']
                ),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
            obj_info.speed = vehicle['speed']
            obj_info.dimensions = Dimensions(
                width=2.1, height=1.5, depth=4.7)  # Typical vehicle dimensions
            msg.objects.append(obj_info)

        # Add closest traffic light if detected
        traffic_light = self.detect_traffic()
        if traffic_light:
            tl_info = TrafficLightsInfo()
            tl_info.id = traffic_light['id']
            tl_info.state = self.traffic_light_state
            tl_info.pose = Pose(
                position=Point(
                    x=traffic_light['position'][0],
                    y=traffic_light['position'][1],
                    z=traffic_light['position'][2]
                ),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
            msg.traffic_lights.append(tl_info)

        msg.safe_to_maneuver = self.safe_to_maneuver()
        self.perception_pub.publish(msg)

    def publish_intersection_flag(self, event):
        self.intersection_flag_pub.publish(Bool(data=self._intersection_flag))

    def publish_safe_in_intersection_flag(self, event):
        """
        Always publish the safe_in_intersection_flag:
        - If the flag is False, publish False immediately and reset buffer.
        - If the flag is True, only publish True after 5 consecutive True detections.
        """
        if not self._safe_in_intersection_flag:
            self._safe_in_intersection_buffer = 0
            self._safe_in_intersection_last_published = False
            self.safe_in_intersection_flag_pub.publish(Bool(data=False))
        else:
            if not self._safe_in_intersection_last_published:
                self._safe_in_intersection_buffer += 1
                if self._safe_in_intersection_buffer >= 5:
                    self._safe_in_intersection_last_published = True
                    self._safe_in_intersection_buffer = 0
                    self.safe_in_intersection_flag_pub.publish(Bool(data=True))
                else:
                    self.safe_in_intersection_flag_pub.publish(Bool(data=False))
            else:
                self.safe_in_intersection_flag_pub.publish(Bool(data=True))

def main():
    try:
        planner_perception = PlannerPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
