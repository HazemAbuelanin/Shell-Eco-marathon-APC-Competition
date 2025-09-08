#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Bool
from std_msgs.msg import Float64
from custom_msg.msg import Perception, ObjectInfo, TrafficLightsInfo, Dimensions
import time

class LoggerNode:
    def __init__(self):
        rospy.init_node('logger_node', anonymous=True)
        
        self.waypoints = np.array([
            [280.363739,-129.306351,0.101746],
            [334.949799,-161.106171,0.001736],
            [339.100037,-258.568939,0.001679],
            [396.295319,-183.195740,0.001678],
            [267.657074,-1.983160,0.001678],
            [153.868896,-26.115866,0.001678],
            [290.515564,-56.175072,0.001677],
            [92.325722,-86.063644,0.001677],
            [88.384346,-287.468567,0.001728],
            [177.594101,-326.386902,0.001677],
            [-1.646942,-197.501282,0.001555],
            [59.701321,-1.970804,0.001467],
            [122.100121,-55.142044,0.001596],
            [161.030975,-129.313187,0.001679],
            [184.758713,-199.424271,0.001680]
        ])
        
        # Initialize waypoint tracking variables
        self.passed_waypoints = set()  # Keep track of passed waypoints
        self.tolerance = 3.0  # 3 meters tolerance
        self.current_position = None
        self.current_mode = None
        self.current_param = None
        self.current_speed = None
        # Initialize distance tracking variables
        self.total_distance = 0.0
        self.last_position = None
        
        # Initialize control command variables
        self.current_throttle = None
        self.current_brake = None
        self.current_steering = None
        
        self.objects = []
        self.traffic_lights = []
        self.state = 4
        
        self.flag = False
        self.safe_to_maneuver = False
        self.high_ways_flag = False
        self.intersection_flag = False
        self.safe_in_intersection_flag = False
        self.start_time = None
        self.end_time = None
        
        # Create subscriber for odometry
        self.odometry_sub = rospy.Subscriber(
            '/carla/ego_vehicle/odometry',
            Odometry,
            self.odometry_callback,
            queue_size=10
        )
        
        rospy.Subscriber(
            '/planner/mode',
            String,
            self.mode_callback
        )
        
        rospy.Subscriber(
            '/carla/ego_vehicle/speedometer',
            Float32,
            self.speedometer_callback
        )
        
        # Add subscribers for control commands
        rospy.Subscriber(
            '/throttle_command',
            Float64,
            self.throttle_callback
        )
        
        rospy.Subscriber(
            '/brake_command',
            Float64,
            self.brake_callback
        )
        
        rospy.Subscriber(
            '/steering_command',
            Float64,
            self.steering_callback
        )
        
        rospy.Subscriber(
            '/planner/perception',
            Perception,
            self.perception_callback
        )
        
        rospy.Subscriber(
            '/planner/flag/change_lane',
            Bool,
            self.flag_callback
        )
        
        rospy.Subscriber(
            '/planner/flag/highway',
            Bool,
            self.high_ways_flag_callback
        )
        
        rospy.Subscriber(
            '/planner/flag/intersection',
            Bool,
            self.intersection_flag_callback
        )
        rospy.Subscriber(
            '/planner/flag/safe_in_intersection',
            Bool,
            self.safe_in_intersection_flag_callback
        )
        
        # Create timer for periodic status updates
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_status)  # Update every second
    
    def odometry_callback(self, msg):
        """Callback for vehicle odometry data"""
        current_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Calculate distance traveled since last position
        if self.last_position is not None:
            distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += distance
        
        self.last_position = current_position
        self.current_position = current_position

        # Remove start_time logic from here!
        # if self.start_time is None and self.current_speed is not None and self.current_speed > 0.1:
        #     self.start_time = rospy.Time.now()

        # Check if we've passed any waypoints
        self.check_waypoints()

    def mode_callback(self, msg):
        """Callback for planner mode data"""
        mode , param = self.parse_mode_string(msg.data)
        self.current_mode = mode
        self.current_param = param
        
    def speedometer_callback(self, msg):
        """Callback for vehicle speedometer data"""
        self.current_speed = msg.data
        # Set start_time only once, when car first moves
        if self.start_time is None and self.current_speed > 0.1:
            self.start_time = time.time()

    def throttle_callback(self, msg):
        """Callback for throttle command data"""
        self.current_throttle = msg.data
        
    def brake_callback(self, msg):
        """Callback for brake command data"""
        self.current_brake = msg.data
        
    def steering_callback(self, msg):
        """Callback for steering command data"""
        self.current_steering = msg.data
    
    def perception_callback(self, msg):
        """Callback for perception data"""
        self.objects = []
        self.traffic_lights = []
        if msg.objects is not None:
            for obj in msg.objects:
                self.objects.append(obj)
        if msg.traffic_lights is not None:
            for tl in msg.traffic_lights:
                self.state = tl.state
                self.traffic_lights.append(tl)
        self.safe_to_maneuver = msg.safe_to_maneuver
    
    def flag_callback(self, msg):
        """Callback for falg data"""
        self.flag = msg.data
                
    def high_ways_flag_callback(self, msg):
        """Callback for high ways flag data"""
        self.high_ways_flag = msg.data 
    
    def intersection_flag_callback(self, msg):
        """Callback for intersection flag"""
        self.intersection_flag = msg.data

    def safe_in_intersection_flag_callback(self, msg):
        """Callback for safe in intersection flag"""
        self.safe_in_intersection_flag = msg.data

    def check_waypoints(self):
        """Check if vehicle has passed any waypoints within tolerance"""
        if self.current_position is None:
            return

        for i, waypoint in enumerate(self.waypoints):
            if i not in self.passed_waypoints:
                distance = np.linalg.norm(self.current_position - waypoint)
                if distance <= self.tolerance:
                    self.passed_waypoints.add(i)
                    # Record end time when reaching the 15th waypoint (index 14)
                    if i == 14 and self.end_time is None:
                        self.end_time = time.time()

    def publish_status(self, event):
        """Periodically publish the current status"""
        if self.current_position is not None:
            rospy.loginfo(f'Passed {len(self.passed_waypoints)} out of {len(self.waypoints)} waypoints')
            rospy.loginfo(f'Total distance traveled: {self.total_distance:.2f} meters out of 2915 meters')
        if self.current_mode is not None:
            if self.current_mode == 'proceed':
                rospy.loginfo(f'Current mode: {self.current_mode}')
            elif self.current_mode == 'stop': 
                rospy.loginfo(f'Current mode: {self.current_mode} with stop distance: {self.current_param}')
            elif self.current_mode == 'follow':
                param = self.current_param * 3.6
                rospy.loginfo(f'Current mode: {self.current_mode} with speed: {param}')
        if self.current_speed is not None:
            rospy.loginfo(f'Current speed: {self.current_speed * 3.6:.2f} km/h')
        if self.current_throttle is not None:
            rospy.loginfo(f'Current throttle: {self.current_throttle:.2f}')
        if self.current_brake is not None:
            rospy.loginfo(f'Current brake: {self.current_brake:.2f}')
        if self.current_steering is not None:
            rospy.loginfo(f'Current steering: {self.current_steering:.2f}')
        if self.objects is not None:
            if self.state == 0:
                state = 'RED'
            elif self.state == 1:
                state = 'YELLOW'
            elif self.state == 2:
                state = 'GREEN'
            else:
                state = 'UNKNOWN'
            rospy.loginfo(f'Perception Status - Objects: {len(self.objects)}, Traffic Lights: {len(self.traffic_lights)}, State: {state}')
        rospy.loginfo(f'Flag (change lane): {self.flag}')
        rospy.loginfo(f'Flag (highways): {self.high_ways_flag}')
        rospy.loginfo(f'Flag (safe to maneuver): {self.safe_to_maneuver}')
        rospy.loginfo(f'Flag (intersection): {self.intersection_flag}')
        rospy.loginfo(f'Flag (safe in intersection): {self.safe_in_intersection_flag}')
        
        # Display live elapsed time using time.time()
        if self.start_time is not None:
            if self.end_time is not None:
                elapsed = self.end_time - self.start_time
                minutes, seconds = divmod(int(elapsed), 60)
                rospy.loginfo(f'Time from start to 15th waypoint: {minutes:02d}:{seconds:02d} (MM:SS)')
            else:
                elapsed = time.time() - self.start_time
                minutes, seconds = divmod(int(elapsed), 60)
                rospy.loginfo(f'Elapsed time since start: {minutes:02d}:{seconds:02d} (MM:SS)')
        rospy.loginfo('-----------------------------------')

    def run(self):
        """Main run loop"""
        rospy.spin()
        
    def parse_mode_string(self, mode_str):
        if ':' in mode_str:
            mode, param = mode_str.split(':')
            try:
                param = float(param)
            except ValueError:
                param = None
            return mode, param
        return mode_str, None

def main():
    logger_node = LoggerNode()
    logger_node.run()
    
if __name__ == '__main__':
    main()
