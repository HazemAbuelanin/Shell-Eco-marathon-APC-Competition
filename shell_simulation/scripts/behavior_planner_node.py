#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32, Bool
from nav_msgs.msg import Odometry
from custom_msg.msg import Perception
import math

# Constants
STATIC_VELOCITY_THRESHOLD = 1.0  # km/h
TARGET_SPEED = 30.0 / 3.6  # 30 km/h
TARGET_SPEED_20 = 20.0 / 3.6  # 20 km/h
TARGET_SPEED_25 = 25.0 / 3.6  # 20 km/h
HIGH_WAY_TARGET_SPEED = 45.0 / 3.6  # 35 km/h


# Object Types
TRAFFIC_LIGHT_TYPE = 1
VEHICLE_TYPE = 2

# Traffic Light States
RED_LIGHT = 0
YELLOW_LIGHT = 1
GREEN_LIGHT = 2
PROCEED_COUNTER = 0
FOLLOW_COUNTER = 0
STOP_COUNTER = 0

REACTION_TIME = 1.5  # seconds
MINIMUM_SAFETY_DISTANCE = 7.0  # meters
DECELERATION_RATE = 7.0  # m/s^2
COMFORTABLE_DECELERATION = 3.0  # m/s^2
SPEED_UPDATE_THRESHOLD = 1  # m/s threshold for speed updates
THW = 2.0  # Time Headway in seconds
TARGET_SPEED_KM = 30.0  # Target speed in m/s
STOP_DISTANCE = 10.0  # Stop distance in meters


class BehaviorTree:
    def __init__(self, planner):
        self.planner = planner
        self.previous_target_speed = 0.0  # Initialize previous target speed
        self.safe_distance = 0.0

    def is_static(self, speed):
        return abs(speed) < STATIC_VELOCITY_THRESHOLD

    def calculate_safe_distance(self, speed):
        """Calculate safe following distance based on current speed"""
        return speed * REACTION_TIME + (speed**2) / (2 * COMFORTABLE_DECELERATION) + MINIMUM_SAFETY_DISTANCE

    def calculate_trarget_speed(self, distance, safe_distance):
        """Calculate target speed using time headway (THW) model"""
        return max(0.0, (distance - abs(distance - safe_distance)) / THW)

    def _should_update_speed(self, new_speed):
        """Check if the speed difference is significant enough to update"""
        return abs(new_speed - self.previous_target_speed) >= SPEED_UPDATE_THRESHOLD

    def execute(self):
        if not self.planner.current_pose:
            return 'stop:0.0'
        
        if self.planner.intersection_flag is False:
            return 'stop:0.0'
        
        # Get closest object
        object_type, closest_object, distance = self.planner.get_closest_object()
        
        if self.planner.flag:
            if object_type == 'vehicle':
                return 'stop:0.0'
            elif object_type == 'traffic_light':
                if closest_object['state'] == RED_LIGHT:
                    return 'stop:0.0'
                elif closest_object['state'] == YELLOW_LIGHT:
                    return 'stop:0.0'
            else:
                if self.planner.safe_to_maneuver:
                    return f'follow:{TARGET_SPEED_20:.2f}'
                else:
                    return 'stop:0.0'


        if object_type is None:
            return self._handle_no_obstacle()
        else:
            return self._handle_obstacle(object_type, closest_object, distance)

    def _handle_no_obstacle(self):
        if self.planner.high_ways_flag:
            return f'follow:{HIGH_WAY_TARGET_SPEED:.2f}'
        else:
            return 'proceed'

    def _handle_obstacle(self, object_type, object_data, distance):
        if object_type == 'traffic_light':
            return self._handle_traffic_light(object_data, distance)
        elif object_type == 'vehicle':
            return self._handle_object(object_data, distance)

    def _handle_traffic_light(self, traffic, distance):
        if traffic['state'] == RED_LIGHT:
            return self._handle_red_light(distance)
        elif traffic['state'] == YELLOW_LIGHT:
            return self._handle_yellow_light(distance)
        elif traffic['state'] == GREEN_LIGHT:
            return f'follow:{TARGET_SPEED_25:.2f}'
        elif traffic['state'] == 4:  
            return f'follow:{TARGET_SPEED_25:.2f}'
        else:  
            return f'follow:{TARGET_SPEED_25:.2f}'

    def _handle_red_light(self, distance):
        stop_distance = max(0.0, distance - 10.0)
        return f'stop:{stop_distance:.2f}'

    def _handle_yellow_light(self, distance):
        # Simple heuristic: if we're close enough and moving fast enough, proceed
        if distance < 10.0 and self.planner.current_speed > 5.0:
            if self.planner.high_ways_flag:
                return f'follow:{HIGH_WAY_TARGET_SPEED:.2f}'
            else:
                return 'proceed'
        else:
            # Stop Before Intersection
            stop_distance = max(0.0, distance - 10.0)
            return f'stop:{stop_distance:.2f}'

    def _handle_object(self, vehicle, distance):
        self.safe_distance = self.calculate_safe_distance(
            self.planner.current_speed)
        front_speed = vehicle['speed']

        if self.is_static(front_speed):
            # Gradually stop before the vehicle
            stop_distance = max(0.0, distance - 10.0)
            return f'stop:{stop_distance:.2f}'

        # CASE 1: We're too far (greater than safe distance)
        if distance > self.safe_distance:
            if self.planner.high_ways_flag:
                return f'follow:{HIGH_WAY_TARGET_SPEED:.2f}'
            else:
                return 'proceed'

        # CASE 2: We're too close
        elif distance < STOP_DISTANCE:
            # Gradually stop before the vehicle
            stop_distance = max(0.0, distance - 10.0)
            return f'stop:{stop_distance:.2f}'
        else:
            new_target_speed = min(
                30 / 3.6, self.calculate_trarget_speed(distance, self.safe_distance))

        # Apply update only if there's a significant change
        if self._should_update_speed(new_target_speed):
            self.previous_target_speed = new_target_speed
            return f'follow:{new_target_speed:.2f}'
        else:
            return f'follow:{self.previous_target_speed:.2f}'


class BehaviorPlannerNode:
    def __init__(self):
        # Initialize state
        self.current_pose = None
        self.current_speed = 0.0
        self.current_object = None
        self.current_distance = float('inf')
        self.flag = False
        self.high_ways_flag = False
        self.safe_to_maneuver = True
        self.intersection_flag = False
        
        # Initialize node
        rospy.init_node('behavior_planner_node', anonymous=True)

        # Add odometry subscriber
        rospy.Subscriber(
            '/carla/ego_vehicle/odometry',
            Odometry,
            self.odometry_callback
        )

        # Add speedometer subscriber
        rospy.Subscriber(
            '/carla/ego_vehicle/speedometer',
            Float32,
            self.speedometer_callback
        )

        # Initialize perception subscriber
        rospy.Subscriber(
            '/planner/perception',
            Perception,
            self.perception_callback
        )

        rospy.Subscriber(
            '/planner/flag/change_lane',
            Bool,
            self.change_lane_flag_callback
        )
        
        
        rospy.Subscriber(
            '/planner/flag/highway',
            Bool,
            self.high_ways_flag_callback
        )
        
        rospy.Subscriber(
            '/planner/flag/safe_in_intersection',
            Bool,
            self.intersection_flag_callback
        )
        
        # Initialize mode publisher
        self.mode_pub = rospy.Publisher(
            '/planner/mode',
            String,
            queue_size=10
        )
        
        
        # Initialize behavior tree
        self.behavior_tree = BehaviorTree(self)

    def odometry_callback(self, msg):
        """Callback for vehicle odometry data"""
        self.current_pose = msg.pose.pose

    def speedometer_callback(self, msg):
        """Callback for vehicle speed data"""
        self.current_speed = msg.data

    def perception_callback(self, msg):
        # Find closest object
        closest_object = None
        closest_distance = float('inf')
        traffic_light_info = None
        object_info = None
        traffic_light_distance = float('inf')
        object_distance = float('inf')

        if self.current_pose is None:
            return

        # Process objects
        for obj in msg.objects:
            object_distance = self.get_distance(
                self.current_pose.position.x,
                self.current_pose.position.y,
                obj.pose.position.x,
                obj.pose.position.y
            )
            object_info = {
                'type': 'vehicle',
                'speed': obj.speed,
                'orientation': {
                    'x': obj.pose.orientation.x,
                    'y': obj.pose.orientation.y,
                    'z': obj.pose.orientation.z,
                    'w': obj.pose.orientation.w
                },
                'location': {
                    'x': obj.pose.position.x,
                    'y': obj.pose.position.y,
                    'z': obj.pose.position.z
                }
            }

        # Process traffic lights
        for tl in msg.traffic_lights:
            traffic_light_distance = self.get_distance(
                self.current_pose.position.x,
                self.current_pose.position.y,
                tl.pose.position.x,
                tl.pose.position.y
            )
            traffic_light_info = {
                'type': 'traffic_light',
                'state': tl.state,
                'location': {
                    'x': tl.pose.position.x,
                    'y': tl.pose.position.y,
                    'z': tl.pose.position.z
                }
            }
        if traffic_light_info or object_info:
            if traffic_light_distance < object_distance:
                closest_object = traffic_light_info
                closest_distance = traffic_light_distance
            else:
                closest_object = object_info
                closest_distance = object_distance
        else:
            closest_object = None
            closest_distance = float('inf')

        # Update current object information
        if closest_object:
            self.current_object = closest_object
            self.current_distance = closest_distance
        else:
            self.current_object = None
            self.current_distance = float('inf')

        # Process the object using behavior tree
        mode = self.behavior_tree.execute()
        safe_distance = self.behavior_tree.safe_distance
        self.safe_to_maneuver = msg.safe_to_maneuver
        self.publish_mode(mode)

    def change_lane_flag_callback(self, msg):
        self.flag = msg.data
    
    def high_ways_flag_callback(self, msg):
        self.high_ways_flag = msg.data
    
    def intersection_flag_callback(self, msg):
        """Callback for intersection flag"""
        self.intersection_flag = msg.data
    
    def get_closest_object(self):
        if self.current_object is None:
            return None, None, float('inf')
        return self.current_object['type'], self.current_object, self.current_distance

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def publish_mode(self, mode):
        """Publish the mode with logic to handle stop and subsequent follow/proceed modes."""
        # Initialize state tracking attributes if not already present
        if not hasattr(self, 'last_mode'):
            self.last_mode = None
        if not hasattr(self, 'follow_proceed_counter'):
            self.follow_proceed_counter = 0

        # Extract the mode type (e.g., 'stop', 'follow', 'proceed')
        mode_type = mode.split(':')[0]

        # If the mode is 'stop', publish it immediately and reset the counter
        if mode_type == 'stop':
            self.last_mode = 'stop'
            self.follow_proceed_counter = 0
            self._publish_mode_message(mode)
            return

        # If the mode is 'follow' or 'proceed' and the last mode was 'stop'
        if self.last_mode == 'stop' and mode_type in ['follow', 'proceed']:
            self.follow_proceed_counter += 1
            if self.follow_proceed_counter >= 5:
                # Allow mode update after 10 consecutive follow/proceed
                self.last_mode = mode_type
                self.follow_proceed_counter = 0
                self._publish_mode_message(mode)
            return

        # For other cases, publish the mode directly
        self.last_mode = mode_type
        self._publish_mode_message(mode)

    def _publish_mode_message(self, mode):
        """Helper function to publish the mode message."""
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)

    def _get_object_type(self, classification):
        # obj type
        if classification == TRAFFIC_LIGHT_TYPE:
            return 'traffic_light'
        elif classification == VEHICLE_TYPE:
            return 'vehicle'
        return None


def main():
    try:
        node = BehaviorPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()