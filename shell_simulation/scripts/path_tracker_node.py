#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
import math
import os
import rospkg
import pandas as pd
from std_msgs.msg import Bool

class PathTracker:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('path_tracker')

        # Parameters
        self.segment_size = rospy.get_param('~segment_size', 50)
        self.min_waypoints = rospy.get_param('~min_waypoints', 10)
        self.offtrack_threshold = rospy.get_param('~offtrack_threshold', 10.0)
        self.flag_distance_threshold = rospy.get_param('~flag_distance_threshold', 15.0)  # Distance threshold for flag detection

        self.waypoints = []  # List to store loaded waypoints
        self.flag_points = []  # List to store flag waypoints (x, y, index)
        self.highway_flag_points = []  # List to store highway flag waypoints (x, y, index)
        self.visited_idx = 0
        self.current_position = (0.0, 0.0)
        self.goal_reached = False
        self.current_flag_point = None  # Currently active flag point
        self.flag_active = False  # Flag status
        self.flag = False  # Flag status for publishing
        self.current_highway_flag_point = None  # Currently active highway flag point
        self.highway_flag_active = False  # Highway flag status
        self.highway_flag = False  # Highway flag status for publishing

        # Load waypoints from CSV
        self.load_waypoints()

        # Publishers
        self.publisher_segment = rospy.Publisher('/planner/trajectory/global', Path, queue_size=10)
        self.publisher_flag = rospy.Publisher('/planner/flag/change_lane', Bool, queue_size=10)
        self.publisher_highway_flag = rospy.Publisher('/planner/flag/highway', Bool, queue_size=10)

        # Subscription for odometry only
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odom_callback, queue_size=10)

        # Timer for flag status publishing
        self.timer_flag = rospy.Timer(rospy.Duration(0.1), self.publish_flag_status)
        self.timer_highway_flag = rospy.Timer(rospy.Duration(0.1), self.publish_highway_flag_status)

    def load_waypoints(self):
        """Load waypoints from CSV file using rospkg and pandas"""
        try:
            # Get package path using rospkg
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('shell_simulation')
            csv_path = os.path.join(package_path, 'data', 'trajectory.csv')
            
            # Read CSV using pandas
            df = pd.read_csv(csv_path)
            x = df.iloc[:, 0]   
            y = df.iloc[:, 1]   
            y = -y
            
            # Check if flag columns exist (4th and 5th columns)
            if len(df.columns) >= 5:
                flags = df.iloc[:, 3]
                highway_flags = df.iloc[:, 4]
                self.waypoints = list(zip(x, y))
                
                # Store flag points
                for i, (x_val, y_val, flag, highway_flag) in enumerate(zip(x, y, flags, highway_flags)):
                    if flag == 1:
                        self.flag_points.append((x_val, y_val, i))
                    if highway_flag == 1:
                        self.highway_flag_points.append((x_val, y_val, i))
            elif len(df.columns) >= 4:
                flags = df.iloc[:, 3]
                self.waypoints = list(zip(x, y))
                for i, (x_val, y_val, flag) in enumerate(zip(x, y, flags)):
                    if flag == 1:
                        self.flag_points.append((x_val, y_val, i))
            else:
                self.waypoints = list(zip(x, y))
            
            if not self.waypoints:
                rospy.logerr("No waypoints loaded from CSV file!")
                
        except Exception as e:
            rospy.logerr(f"Error loading waypoints: {str(e)}")
            self.waypoints = []  # Initialize empty waypoints list

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        if not self.waypoints or self.goal_reached:
            return

        # Find nearest waypoint
        self.visited_idx = self.find_nearest_point(
            self.current_position,
            start_idx=max(0, self.visited_idx - 10)
        )

        # Publish path segment
        self.publish_path_segment()
        
        # Check flag points
        self.check_flag_status()
        self.update_highway_flag_status()

    def check_flag_status(self):
        """Check if we are near any flag point and update flag status"""
        # If no flag is active, check if we're approaching one
        if not self.flag_active:
            for x, y, idx in self.flag_points:
                dist = self.distance(self.current_position, (x, y))
                if dist <= self.flag_distance_threshold:
                    self.current_flag_point = (x, y, idx)
                    self.flag_active = True
                    self.flag = True
                    break
        
        # If flag is active, check if we're moving away from it
        elif self.current_flag_point:
            x, y, _ = self.current_flag_point
            dist = self.distance(self.current_position, (x, y))
            if dist > self.flag_distance_threshold:
                self.flag_active = False
                self.current_flag_point = None
                self.flag = False

    def check_highway_flag_status(self):
        """Check if we are near any highway flag point and update highway flag status"""
        # If no highway flag is active, check if we're approaching one
        if not self.highway_flag_active:
            for x, y, idx in self.highway_flag_points:
                dist = self.distance(self.current_position, (x, y))
                if dist <= self.flag_distance_threshold:
                    self.current_highway_flag_point = (x, y, idx)
                    self.highway_flag_active = True
                    self.highway_flag = True
                    break
        # If highway flag is active, check if we're moving away from it
        elif self.current_highway_flag_point:
            x, y, _ = self.current_highway_flag_point
            dist = self.distance(self.current_position, (x, y))
            if dist > self.flag_distance_threshold:
                self.highway_flag_active = False
                self.current_highway_flag_point = None
                self.highway_flag = False

    def update_highway_flag_status(self):
        """Set highway flag True if the closest waypoint has the 5th field (new_flag) == 1, else False."""
        try:
            # Read the CSV again to get the new_flag value for the closest waypoint
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('shell_simulation')
            csv_path = os.path.join(package_path, 'data', 'trajectory.csv')
            df = pd.read_csv(csv_path)
            if len(df.columns) >= 5:
                new_flag = df.iloc[self.visited_idx, 4]
                self.highway_flag = bool(new_flag)
            else:
                self.highway_flag = False
        except Exception as e:
            rospy.logerr(f"Error updating highway flag: {str(e)}")
            self.highway_flag = False

    def publish_flag_status(self, event=None):
        """Publish the flag status as a Boolean message"""
        msg = Bool()
        msg.data = self.flag
        self.publisher_flag.publish(msg)

    def publish_highway_flag_status(self, event=None):
        """Publish the highway flag status as a Boolean message"""
        msg = Bool()
        msg.data = self.highway_flag
        self.publisher_highway_flag.publish(msg)

    def find_nearest_point(self, position, start_idx=0):
        min_dist = float('inf')
        best_idx = start_idx
        search_end = min(start_idx + 100, len(self.waypoints))
        
        for idx in range(start_idx, search_end):
            dist = self.distance(position, self.waypoints[idx])
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                
        return best_idx

    def publish_path_segment(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        start_idx = self.visited_idx
        end_idx = min(start_idx + self.segment_size, len(self.waypoints))
        
        # Ensure minimum waypoints are published
        if (end_idx - start_idx) < self.min_waypoints:
            end_idx = min(start_idx + self.min_waypoints, len(self.waypoints))
            if (end_idx - start_idx) < self.min_waypoints:
                rospy.logwarn("Cannot publish sufficient waypoints!")
                return

        for idx in range(start_idx, end_idx):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = self.waypoints[idx][0]
            pose.pose.position.y = self.waypoints[idx][1]
            path_msg.poses.append(pose)

        self.publisher_segment.publish(path_msg)

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    try:
        path_tracker = PathTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main() 
